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

mod scratch_support;
use scratch_support::Scratch;

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

/// `src` with every comment blanked, string and char literals kept.
///
/// Byte for byte as long as `src`. Literals stay because a marker inside a
/// string is the conservative side for the device scan; comments go because a
/// marker named in prose opens nothing. Comments are found by the lexer in
/// `literal_end`, not by cutting at the first `//`, which a `https://` inside a
/// string also has (that cut dropped the rest of the line, quotes and braces
/// included, and could leave a body unterminated).
fn blank_comments(src: &str) -> String {
    blank_tokens(src, false)
}

/// Where each `fn name` definition starts in `blank` (literal-blanked source):
/// `fn`, then whitespace of any kind and length, then exactly `name`, then
/// optional whitespace and `(` or `<`. So `fn helper<T>(`, `fn helper (`,
/// `fn  helper(` and `fn\thelper(` are definitions, `fn helper_two(` is not, and
/// the `fn(u8)` of a fn-pointer type is not.
fn fn_sites(blank: &str, name: &str) -> Vec<usize> {
    blank
        .match_indices("fn")
        .map(|(i, _)| i)
        .filter(|&i| {
            at_word_start(blank, i)
                && declared_name(blank, i, "fn").is_some_and(|(found, after)| {
                    let rest = blank[after..].trim_start();
                    found == name && (rest.starts_with('(') || rest.starts_with('<'))
                })
        })
        .collect()
}

/// Does `body` call `callee`: the name at a word boundary, then `(` or a
/// turbofish (`callee::<T>(`), whitespace allowed before either.
fn calls(body: &str, callee: &str) -> bool {
    body.match_indices(callee).any(|(i, _)| {
        let rest = body[i + callee.len()..].trim_start();
        at_word_start(body, i) && (rest.starts_with('(') || rest.starts_with("::<"))
    })
}

/// How the signature that starts at `at` ends.
enum SignatureEnd {
    /// The `{` that opens the body, at this offset.
    Body(usize),
    /// A `;` first: a declaration with no body.
    Declaration,
    /// Neither: unbalanced generics, or the text ran out.
    Unterminated,
}

/// Where the signature of the `fn` whose keyword is at `at` ends.
///
/// The body is the first `{` that is outside every parenthesis and bracket and
/// outside every `<...>`. A `{` inside angle brackets is a const-generic
/// expression (`Tr<{ 1 }>`, `Foo<{ N >= 3 }>`, a default `<const N: usize = { 3 }>`),
/// and is skipped whole, braces and anything inside them: a `>` in there is an
/// operator, not a closing bracket. In the signature proper, a `>` closes an
/// angle bracket unless it is the head of `->`, and `>>` and `>=` close as many
/// as they hold. Angle depth is only tracked outside `()` and `[]`, which cover
/// the other places `<` and `>` are comparisons.
///
/// A `;` while an angle bracket is still open is `Unterminated`, not a
/// declaration: an unbalanced `<` would otherwise swallow the body and read the
/// next item's `;` as this function's end, a silent pass.
fn signature_end(blank: &str, at: usize) -> SignatureEnd {
    let (mut nesting, mut angle, mut braces) = (0usize, 0usize, 0usize);
    let mut prev = ' ';
    for (i, c) in blank[at..].char_indices() {
        if braces > 0 {
            match c {
                '{' => braces += 1,
                '}' => braces -= 1,
                _ => {}
            }
            continue;
        }
        match c {
            '(' | '[' => nesting += 1,
            ')' | ']' => nesting = nesting.saturating_sub(1),
            '<' if nesting == 0 => angle += 1,
            '>' if nesting == 0 && prev != '-' => angle = angle.saturating_sub(1),
            ';' if nesting == 0 && angle > 0 => return SignatureEnd::Unterminated,
            ';' if nesting == 0 => return SignatureEnd::Declaration,
            '{' if nesting == 0 && angle > 0 => braces = 1,
            '{' if nesting == 0 => return SignatureEnd::Body(at + i),
            _ => {}
        }
        prev = c;
    }
    SignatureEnd::Unterminated
}

/// The `{...}` body of the first function named `name` in `src` that has one.
/// `Ok(None)` if `src` defines none, or only declares it (`fn name(..);` in a
/// trait). `Err` names the function if its body cannot be delimited. There is no
/// guessed span: the callers are checks, and a check that cannot read a body
/// must say so rather than scan whatever text happens to follow it.
///
/// A bodiless declaration is skipped, not taken as the answer: with a trait
/// `fn go(&self);` ahead of `impl T for X { fn go(&self) { .. } }`, the impl is
/// the body that is scanned.
///
/// Structure is read off the literal-blanked copy, so a definition is never a
/// `fn name(` quoted inside a string (a pin that asserts on another function's
/// source holds exactly that), and a `{` or `}` in a string, raw string, byte or
/// C string, char literal or comment is not counted. Spans are taken from `src`
/// itself, which `blank_literals` keeps offset-aligned. Where the signature ends
/// is `signature_end`'s call: a `{` inside generics, a where clause or a return
/// type is not the body.
///
/// Known gaps, all pre-existing:
/// - resolution is by name alone. When several functions share a name (six
///   `fn new(` in `src/runtime.rs`), the first one with a body is the one
///   scanned, whichever type a call actually reaches. Narrowing by receiver or
///   path is not done here, and scanning every same-named body would flag
///   device-free tests that merely call some other type's `new`. (Resolving
///   `Type::name(` to `impl Type` exists on branch `gp/w10-scangaps-item1` and
///   was not landed: against 936fe8e it flips 30 lib tests of
///   `runtime::pool_ledger_tests` from pass to flag, none of them in the
///   manifest. 13 of the 30 were traced to `Rig::new`, which calls
///   `GpuRuntime::new`; the other 17 were not traced.)
/// - a helper defined by `macro_rules!` is not a `fn name` in the source text, so
///   its body is never scanned: the scan reads source, not expansions;
/// - a function reached through a pointer (`let f = make; f();`) is not a call
///   to `make` in the text, so it is not followed. Following every bare mention
///   of a defined name was measured against all 148 exact entries and 415 library
///   tests: it catches nothing the call walk misses, and flags one test that opens no device,
///   `runtime::mid_commit_tests::metal4_encode_survives_a_mid_commit_at_every_dispatch`,
///   which only spawns a child process.
fn fn_body<'a>(src: &'a str, name: &str) -> Result<Option<&'a str>, String> {
    let blank = blank_literals(src);
    for at in fn_sites(&blank, name) {
        let open = match signature_end(&blank, at) {
            SignatureEnd::Declaration => continue,
            SignatureEnd::Body(open) => open,
            SignatureEnd::Unterminated => return Err(format!("`fn {name}`: no `{{` after the signature")),
        };
        let mut depth = 0usize;
        let mut end = None;
        for (i, c) in blank[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1; // `open` is a `{`, so depth is at least 1 here
                    if depth == 0 {
                        end = Some(open + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.ok_or_else(|| {
            format!("`fn {name}`: its body opens at byte {open} and never closes ({depth} `{{` left open)")
        })?;
        return Ok(Some(&src[open..end]));
    }
    Ok(None)
}

/// Every function name defined in `src`: `fn`, whitespace of any kind, a name.
fn defined_fns(src: &str) -> HashSet<String> {
    src.match_indices("fn")
        .filter_map(|(i, _)| declared_name(src, i, "fn").map(|(name, _)| name))
        .collect()
}

/// `body` with every `DEVICE_MARKER_EXEMPT` call path replaced, so the marker
/// scan does not read a device type's name out of a call that opens nothing.
/// Nothing else is touched: a mention of the type in any other form still
/// counts, which is what keeps this narrower than retiring the marker.
fn without_exempt_calls(body: &str) -> String {
    let mut out = body.to_string();
    for (path, _, _) in DEVICE_MARKER_EXEMPT {
        // Only where the path is the whole callee, i.e. followed directly by
        // `(`. A plain `replace` is a prefix match: it would also blank the
        // front of a longer name — `…is_event_wait_timeout_and_opens()` — and
        // hide a device call behind an exemption that never covered it. Anything
        // else, a bare reference included, is left to be flagged.
        let mut from = 0;
        while let Some(i) = out[from..].find(path) {
            let at = from + i;
            let after = at + path.len();
            if out[after..].starts_with('(') {
                out.replace_range(at..after, "exempt_pure_call");
                from = at + "exempt_pure_call".len();
            } else {
                from = after;
            }
        }
    }
    out
}

/// A device marker reachable from `test` through functions defined in `src`.
fn device_marker(src: &str, test: &str) -> Result<Option<String>, String> {
    let src = blank_comments(src);
    let known = defined_fns(&src);
    let mut seen = HashSet::new();
    let mut work = vec![test.to_string()];
    while let Some(name) = work.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(body) = fn_body(&src, &name)? else {
            if name == test {
                return Err(format!("no `fn {test}(` with a body found"));
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
            if calls(body, callee) {
                work.push(callee.clone());
            }
        }
    }
    Ok(None)
}

/// The file a manifest target lives in: `tests/BIN.rs`, or for `lib` the file
/// the library's module tree (`lib`, from `lib_tests_in`) declares the test in.
///
/// A `lib` name is resolved through the module walk the `lib` filters use, not
/// spelled from its path: a `#[path = "…"] mod` puts a test in a file its module
/// path does not name (`runtime::pool_ledger_tests` lives in
/// `src/pool_ledger_tests.rs`, not `src/runtime.rs`).
fn source_of(root: &Path, lib: &[LibTest], target: &str, name: &str) -> PathBuf {
    if target != "lib" {
        return root.join("tests").join(format!("{target}.rs"));
    }
    lib_test_named(lib, name).file.clone()
}

/// The library `#[test]` whose libtest path is exactly `name`. A name the walk
/// does not reach (misspelled, `#[ignore]`d, or behind a feature the hosted job
/// does not build) is refused: the hosted job would run nothing for it.
fn lib_test_named<'a>(lib: &'a [LibTest], name: &str) -> &'a LibTest {
    lib.iter().find(|t| t.path == name).unwrap_or_else(|| {
        panic!(
            "lib {name}: no #[test] at that path in the library's module tree \
             (misspelled, #[ignore]d, or behind a feature the hosted job does not build)"
        )
    })
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
/// `blank_comments` is there for — it cuts at the first `//`, which a `https://`
/// inside a string literal also has.
///
/// Byte for byte as long as `src`, so an offset into one indexes the other: the
/// structure is read off this, and an attribute's own text — a `#[path = "…"]`
/// — off the source it came from.
fn blank_literals(src: &str) -> String {
    blank_tokens(src, true)
}

/// `blank_literals`, or with `strings` false only the comments blanked.
fn blank_tokens(src: &str, strings: bool) -> String {
    let c: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < c.len() {
        let end = literal_end(&c, i);
        if end > i && (strings || c[i] == '/') {
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
        } else if end > i {
            out.extend(&c[i..end]);
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
        // `b"…"`, `c"…"`, `r"…"`, `br"…"`, `cr"…"` and their `#`-hashed raw forms. A name
        // character in front means this is the tail of an identifier.
        'b' | 'c' | 'r' if i == 0 || !name_char(c[i - 1]) => {
            let mut j = i;
            if at(j) == 'b' || at(j) == 'c' {
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
        // A char literal, against a lifetime: `'a'` against `'a`. Every escape
        // form: `'\n'`, `'\''`, `'\\'`, `'\x41'` (two hex digits), `'\u{7b}'`.
        '\'' if at(i + 1) == '\\' => {
            let mut j = i + 3;
            match at(i + 2) {
                'x' => j = i + 5,
                'u' if at(i + 3) == '{' => {
                    j = i + 4;
                    while j < c.len() && at(j) != '}' {
                        j += 1;
                    }
                    j += 1;
                }
                _ => {}
            }
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
    fn_sites(src, name).len()
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
    let Some(&at) = fn_sites(&blank_literals(src), name).first() else {
        return false;
    };
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
    let lib = lib_tests_in(root());
    for (target, name, ignored) in every_name() {
        let leaf = name.rsplit("::").next().unwrap();
        let path = source_of(root(), &lib, &target, &name);
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
///
/// An `exact lib` name is scanned through `scan_resolved`, the same as a test a
/// `lib` filter selects, so both reach its file through the module walk and both
/// refuse a leaf name that file defines twice.
fn check_exact_device_free(manifest: &str, root: &Path) {
    let lib = lib_tests_in(root);
    let mut cache = HashMap::new();
    for e in parse(manifest) {
        let Entry::Exact { target, names } = e else { continue };
        for name in names {
            if target == "lib" {
                scan_resolved(lib_test_named(&lib, &name), &mut cache, "an `exact lib` entry");
                continue;
            }
            let leaf = name.rsplit("::").next().unwrap();
            let path = source_of(root, &lib, &target, &name);
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
    let dir = Scratch::dir("tessl-hosted-selection-planted");
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(
        dir.join("src/lib.rs"),
        "pub mod planted;\npub mod clean;\npub mod exempt_edge;\npub mod host;\n",
    )
    .unwrap();
    // A `#[path]` test module, the shape of `runtime::pool_ledger_tests`: its
    // tests live in a file the module path `host::host_tests` does not spell.
    fs::write(
        dir.join("src/host.rs"),
        "#[cfg(test)]\n#[path = \"host_tests.rs\"]\nmod host_tests;\n",
    )
    .unwrap();
    fs::write(
        dir.join("src/host_tests.rs"),
        concat!(
            "#[test]\nfn sums() {\n    assert_eq!(3 + 3, 6);\n}\n",
            "#[test]\nfn reaches_the_device() {\n    with_gpu(|rt| drop(rt));\n}\n",
        ),
    )
    .unwrap();
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
        concat!(
            "#[cfg(test)]\nmod tests {\n",
            "    #[test]\n    fn adds_up() {\n        assert_eq!(2 + 2, 4);\n    }\n",
            // The exempt call itself, so the exemption is shown to fire.
            "    #[test]\n    fn calls_the_exempt_pure_function() {\n",
            "        assert!(!GpuRuntime::is_event_wait_timeout(\"x\"));\n    }\n",
            "}\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("src/exempt_edge.rs"),
        concat!(
            // A longer name sharing the exempt path's prefix. The exemption must
            // not reach it: a prefix match here would blank `GpuRuntime` and
            // wave a real device call through.
            "#[cfg(test)]\nmod tests {\n",
            "    #[test]\n    fn a_longer_name_sharing_the_prefix_is_not_exempt() {\n",
            "        let _ = GpuRuntime::is_event_wait_timeout_and_then_opens_one(\"x\");\n    }\n",
            "}\n",
        ),
    )
    .unwrap();

    // The resolver reads the layout, and leaves the `#[ignore]`d test out.
    assert_eq!(
        lib_tests_in(&dir).iter().map(|t| t.path.as_str()).collect::<Vec<_>>(),
        [
            "clean::tests::adds_up",
            "clean::tests::calls_the_exempt_pure_function",
            "exempt_edge::tests::a_longer_name_sharing_the_prefix_is_not_exempt",
            "host::host_tests::reaches_the_device",
            "host::host_tests::sums",
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
    check_lib_device_free("lib clean::tests:: 2\n", &dir);

    // The exemption fires for the exact call form and reaches no further. Both
    // halves matter: without the first, `lib shared::tests::` is refused for a
    // test that opens nothing; without the second, the exemption is a prefix
    // match and a longer name that *does* open a device rides in behind it.
    assert!(
        refused("lib exempt_edge::tests:: 1\n", check_lib_device_free),
        "a longer name sharing the exempt path's prefix was treated as exempt, \
         which would hide a real device call"
    );

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

    // The `exact lib` arm, into a `#[path]` module. Resolving the file by
    // spelling the module path (`src/host.rs`) finds no `fn sums(` there and
    // refuses a clean test; it has to come from the module walk.
    check_exact_device_free("exact lib host::host_tests::sums\n", &dir);
    assert!(
        refused(
            "exact lib host::host_tests::reaches_the_device\n",
            check_exact_device_free
        ),
        "an `exact lib` entry naming a device-opening test in a #[path] module was reported clean"
    );
    assert!(
        refused("exact lib host::host_tests::not_a_test\n", check_exact_device_free),
        "an `exact lib` entry naming no test was accepted"
    );
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
        let stripped = blank_comments(&src);

        // One definition, or the exemption would rest on whichever body came first.
        // Counted over literal-blanked source, which cannot mistake a `fn x(`
        // inside a string for a definition; the marker check below stays on
        // `blank_comments` output, both to match what `device_marker` itself
        // scans and because keeping string contents is the conservative side
        // for a check whose job is to refuse the exemption.
        let defined = fn_definitions(&blank_literals(&src), name);
        assert_eq!(
            defined,
            1,
            "{path}: {} defines `fn {name}(` {defined} time(s), so which body the \
             exemption rests on is not decidable",
            full.display()
        );

        let body = fn_body(&stripped, name)
            .unwrap_or_else(|e| panic!("{path}: {e} in {}", full.display()))
            .unwrap_or_else(|| panic!("{path}: no `fn {name}` with a body in {}", full.display()));
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

/// `fn_body` unwrapped, for the self-tests below.
fn body_of<'a>(src: &'a str, name: &str) -> &'a str {
    fn_body(src, name)
        .unwrap_or_else(|e| panic!("{e}"))
        .unwrap_or_else(|| panic!("no body for {name}"))
}

/// Braces in every literal and comment kind, none of which may move the count.
/// The raw strings use `##` so a `"#` inside them is not their end.
#[test]
fn fn_body_ignores_braces_in_literals_and_comments() {
    let src = r####"
fn sig_comment(x: u8) /* { not the body */ -> u8 // { nor this
{
    let a = "open { and quote \" and close }";
    let b = r##"raw { "# still raw } "##;
    let c = b"byte {";
    let d = br#"byte raw } "#;
    let (e, f, g, h) = ('{', '}', '\'', b'}');
    let i = '\u{7b}';
    fn lifetime<'a>(s: &'a str) -> &'a str { s }
    // a line comment with { and a https://x.y/z URL
    /* a block comment with } /* nested { */ still a comment } */
    let url = "https://example.com/{";
    drop((a, b, c, d, e, f, g, h, i, url));
    0
}
fn after() { let _rt = GpuRuntime::new(); }
"####;
    let body = body_of(src, "sig_comment");
    assert!(body.starts_with("{\n    let a = \"open"), "body starts wrong: {body:?}");
    assert!(body.ends_with("    0\n}"), "body ends wrong: {body:?}");
    assert!(!body.contains("GpuRuntime"), "the body ran on into the next function");
    assert_eq!(body_of(src, "after"), "{ let _rt = GpuRuntime::new(); }");
}

/// A body that cannot be delimited is an error naming the function, never a span
/// guessed from the text after it.
#[test]
fn fn_body_refuses_a_body_it_cannot_delimit() {
    let err = fn_body("fn open_ended() { let a = 1;\n#[test]\nfn next() {}\n", "open_ended").unwrap_err();
    assert!(err.contains("open_ended") && err.contains("never closes"), "{err}");
    let err = fn_body("fn headless(x: u8)", "headless").unwrap_err();
    assert!(err.contains("headless"), "{err}");
    // A declaration has no body to scan, and must not borrow the next item's.
    assert_eq!(fn_body("fn decl(&self);\nfn other() { 1 }\n", "decl"), Ok(None));
    assert_eq!(fn_body("fn other() {}", "absent"), Ok(None));
    // `device_marker` surfaces it rather than reporting the test clean.
    assert!(device_marker("fn t() { let a = 1;", "t").is_err());
}

/// The 3a83bda layout. A pin asserts on another function's source, so its body
/// quotes `fn end_command_buffer(` and a `{` that never closes. The old scanner
/// took the quoted text for a definition, counted from the quoted `{`, never
/// closed, and ran on to the next `#[test]`: GpuRuntime tests appended after the
/// pin were read as the pin's own code and the exact entry was refused. It must
/// pass because the pin's body ends at its own brace, so the span is asserted.
#[test]
fn a_pin_that_quotes_a_function_is_not_charged_for_the_tests_after_it() {
    let src = r#"
#[test]
fn pin() {
    let ender = compact_fn(src, "fn end_command_buffer(");
    let want = compact("fn end_command_buffer(&self) { self.endCommandBuffer();");
    assert_eq!(ender, want);
}

#[test]
fn appended_after_the_pin() {
    let rt = GpuRuntime::new();
    drop(rt);
}
"#;
    assert_eq!(
        fn_body(src, "end_command_buffer"),
        Ok(None),
        "a quoted `fn x(` was read as a definition"
    );
    assert_eq!(
        device_marker(src, "pin"),
        Ok(None),
        "the pin was charged for the test after it"
    );
    assert!(device_marker(src, "appended_after_the_pin").unwrap().is_some());
    let pin = body_of(src, "pin");
    assert!(pin.ends_with("assert_eq!(ender, want);\n}"), "span: {pin:?}");
}

/// The false pass the old count allowed: a `}` in a string closed the body
/// early, and a device use after it went unseen.
#[test]
fn a_device_use_after_a_brace_in_a_literal_is_still_flagged() {
    for (label, lead) in [
        ("string", r#"let s = "}";"#),
        ("raw string", r##"let s = r#"} }"#;"##),
        ("char", "let s = '}';"),
        ("byte char", "let s = b'}';"),
        ("byte string", r#"let s = b"}}";"#),
        ("escaped quote", r#"let s = "\"}";"#),
        ("url before a brace", r#"let s = "https://x/}"; let t = "}";"#),
    ] {
        let src = format!("#[test]\nfn pinned() {{\n    {lead}\n    let _rt = GpuRuntime::new();\n}}\n");
        let found = device_marker(&src, "pinned").unwrap();
        assert!(
            found.is_some_and(|m| m.contains("GpuRuntime")),
            "{label}: a GpuRuntime use after a brace in a literal went unseen"
        );
    }
    // A `//` inside a string is not a comment: what follows it on the line is code.
    let src = "#[test]\nfn pinned() { let u = \"https://x\"; let _rt = GpuRuntime::new(); }\n";
    assert!(
        device_marker(src, "pinned").unwrap().is_some(),
        "a `//` in a string hid the rest of the line"
    );
    // And a marker named only in a comment still does not count.
    let src = "#[test]\nfn fine() { let u = \"}\"; /* with_gpu } */ // GpuRuntime\n let _ = u; }\n";
    assert!(device_marker(src, "fine").unwrap().is_none());
}

/// A trait declaration comes before its impls. The declaration has no body, and
/// must not stand in for the definitions: the old scanner took the first
/// `fn go(`, found nothing to scan, and a device call in an impl went unseen.
/// Every body of the name is scanned, because the scan cannot tell which impl a
/// call reaches.
#[test]
fn a_bodyless_declaration_does_not_hide_the_impl_that_follows() {
    let src = "trait T { fn go(&self); }\n\
               impl T for B { fn go(&self) { let _rt = GpuRuntime::new(); } }\n\
               #[test]\nfn t() { A.go(); }\n";
    assert_eq!(
        body_of(src, "go"),
        "{ let _rt = GpuRuntime::new(); }",
        "the declaration was taken as the definition, or a later impl was preferred"
    );
    assert!(
        device_marker(src, "t")
            .unwrap()
            .is_some_and(|m| m.contains("go mentions GpuRuntime")),
        "the impl body that follows the declaration went unscanned"
    );
    // A name only ever declared has no body to scan and is not an error.
    let src = "trait T { fn go(&self); }\n#[test]\nfn t() { A.go(); }\n";
    assert_eq!(device_marker(src, "t"), Ok(None));
}

/// Char escapes other than `\n`, `\'`, `\\` and `\u{..}`: `'\x41'` was not
/// read as a literal, so the `'` after it opened a bogus one and swallowed a real
/// brace, closing the body early.
#[test]
fn every_char_escape_form_is_one_literal() {
    for lit in ["'\\n'", "'\\x41'", "'\\u{7b}'", "'\\''", "'\\\\'", "'\\0'", "b'\\x7d'"] {
        // No space after the comma: `',` followed by `'` is what a lexer that
        // missed the escape reads as a char literal `','`, exposing the brace.
        let src = format!("#[test]\nfn t() {{ let c = ({lit},'}}'); let _rt = GpuRuntime::new(); }}\n");
        assert!(
            device_marker(&src, "t").unwrap().is_some(),
            "{lit} was not read as a single literal: the body closed early"
        );
        let src = format!("fn t() {{ let c = ({lit},'}}'); let z = 1; }}\nfn u() {{ }}\n");
        assert!(
            body_of(&src, "t").ends_with("let z = 1; }"),
            "{lit}: the span of `t` is wrong"
        );
    }
}

/// C-string literals (`c"…"`, `cr#"…"#`) hold braces like any other string.
#[test]
fn c_strings_are_literals() {
    // The hashed raw forms hold a `"` inside, so a lexer that does not know the
    // `c` prefix ends the string at that quote and exposes the brace after it.
    for lit in [
        r#"c"}""#,
        r##"cr#"x"} "#"##,
        r###"cr##"x"# }"##"###,
        r#"c"\"}""#,
        r#"cr"}""#,
    ] {
        let src = format!("#[test]\nfn t() {{ let s = {lit}; let _rt = GpuRuntime::new(); }}\n");
        assert!(
            device_marker(&src, "t").unwrap().is_some(),
            "{lit}: a brace in a C string closed the body early"
        );
    }
}

/// Helpers the call walk could not reach: generic (`fn helper<T>(`), spaced
/// (`fn helper (`), called through a turbofish.
#[test]
fn generic_and_spaced_helpers_are_scanned() {
    for (label, def, call) in [
        (
            "generic",
            "fn helper<T: Copy>(x: T) { let _rt = GpuRuntime::new(); }",
            "helper(1);",
        ),
        ("spaced", "fn helper () { let _rt = GpuRuntime::new(); }", "helper();"),
        (
            "turbofish",
            "fn helper<T>() { let _rt = GpuRuntime::new(); }",
            "helper::<u8>();",
        ),
        (
            "spaced call",
            "fn helper() { let _rt = GpuRuntime::new(); }",
            "helper ();",
        ),
    ] {
        let src = format!("{def}\n#[test]\nfn t() {{ {call} }}\n");
        assert!(
            device_marker(&src, "t").unwrap().is_some(),
            "{label}: the helper's device call went unseen"
        );
    }
    // A longer name is not the helper: `fn helper_two(` is not `fn helper`.
    let src = "fn helper_two() { let _rt = GpuRuntime::new(); }\n#[test]\nfn t() { helper(); }\n";
    assert_eq!(device_marker(src, "t"), Ok(None));
}

/// A device marker inside a string literal counts. A string can be `format!`ed
/// into a command that opens a device, and the scan cannot tell, so it takes the
/// conservative side. `blank_comments` must blank comments only: if it ever
/// blanked strings too, this goes red.
#[test]
fn a_marker_inside_a_string_literal_still_counts() {
    let src = "#[test]\nfn t() { let s = \"GpuRuntime\"; drop(s); }\n";
    assert!(device_marker(src, "t")
        .unwrap()
        .is_some_and(|m| m.contains("GpuRuntime")));
    let src = "#[test]\nfn t() { let s = r#\"with_gpu\"#; drop(s); }\n";
    assert!(device_marker(src, "t").unwrap().is_some());
    assert_eq!(
        blank_comments("let s = \"a // b\"; // c\n"),
        "let s = \"a // b\";     \n"
    );
}

/// A quoted definition that is followed by a block, `"fn x() { }"`, is as good
/// a definition as the real one to a lookup on raw source. The real `x` is
/// later and opens a device; the walk must reach it, not the quoted empty one.
#[test]
fn a_quoted_definition_with_a_body_does_not_shadow_the_real_one() {
    // The quoted `fn x() { }` is followed, with no `;` between, by the `{` of an
    // `if`: a lookup that took the quoted text for a definition would read that
    // block as `x`'s body, and the real `x` below would go unscanned.
    let src = "#[test]\nfn pin() { if \"fn x() { }\".is_empty() { let _ = 1; } x(); }\n\
               fn x() { let _rt = GpuRuntime::new(); }\n";
    assert_eq!(body_of(src, "x"), "{ let _rt = GpuRuntime::new(); }");
    assert!(device_marker(src, "pin")
        .unwrap()
        .is_some_and(|m| m.contains("x mentions GpuRuntime")));
}

/// A `{` that is not the body, before the body. Generics, a where clause and a
/// return type can all hold a const-generic block; the old signature scan took
/// the first `{` outside `()` and `[]` and read that block as the function.
#[test]
fn a_brace_inside_generics_is_not_the_body() {
    for (label, def) in [
        (
            "where T: Tr<{1}> (P24)",
            "fn h<T>() where T: Tr<{1}> { let _rt = GpuRuntime::new(); }",
        ),
        (
            "where on a non-generic fn (P24c)",
            "fn h() where u8: Tr<{1}> { let _rt = GpuRuntime::new(); }",
        ),
        (
            "const default (P25)",
            "fn h<const N: usize = { 3 }>() { let _rt = GpuRuntime::new(); }",
        ),
        (
            "return type (P6)",
            "fn h() -> Foo<{ 3 }> { let _rt = GpuRuntime::new(); }",
        ),
        (
            "an operator in the block",
            "fn h() -> Foo<{ N >= 3 && N > 2 }> { let _rt = GpuRuntime::new(); }",
        ),
        (
            "a nested block",
            "fn h() where T: Tr<{ { 1 } }> { let _rt = GpuRuntime::new(); }",
        ),
        (
            "a `->` inside the generics, then a block",
            "fn h<F: Fn() -> u8, const N: usize = { 3 }>() { let _rt = GpuRuntime::new(); }",
        ),
        (
            "arrow, closing pair and a block",
            "fn h<F: Fn(u8) -> Vec<Vec<u8>>>() -> impl Fn() -> Foo<{ 2 }> { let _rt = GpuRuntime::new(); }",
        ),
    ] {
        // As a helper the test calls, and as the test itself.
        let src = format!("{def}\n#[test]\nfn t() {{ h(); }}\n");
        let body = body_of(&src, "h");
        assert!(
            body.contains("GpuRuntime") && body.starts_with("{ let"),
            "{label}: the body was read as {body:?}"
        );
        assert!(
            device_marker(&src, "t").unwrap().is_some(),
            "{label}: the device call in the helper went unseen"
        );
        let as_test = def.replacen("fn h", "fn t", 1);
        assert!(
            device_marker(&as_test, "t").unwrap().is_some(),
            "{label}: the device call in the test itself went unseen"
        );
    }
}

/// The other side: angle tracking must not turn a clean signature into a flag or
/// an error, and must not eat a body that follows a `->`, a `>>` or a comparison.
#[test]
fn angle_tracking_keeps_clean_signatures_clean() {
    for def in [
        "fn t() -> Vec<Vec<u8>> { vec![] }",
        "fn t<F: Fn(u8) -> u8>(f: F) -> u8 { f(1) }",
        "fn t() -> impl Fn() -> u8 { || 1 }",
        "fn t<const N: usize>() -> [u8; N] where [(); N * 2]: { [0; N] }",
        "fn t(a: u8, b: u8) -> bool where u8: Tr<{ 1 < 2 }> { a < b && b > a }",
        "fn t<'a>(s: &'a str) -> Result<&'a str, ()> { Ok(s) }",
    ] {
        let src = format!("{def}\nfn after() {{ let _rt = GpuRuntime::new(); }}\n");
        assert_eq!(device_marker(&src, "t"), Ok(None), "{def}: the scan ran past the body");
        assert!(
            body_of(&src, "t").ends_with('}') && !body_of(&src, "t").contains("GpuRuntime"),
            "{def}: wrong span"
        );
    }
    // Unbalanced generics are an error, never a quiet skip of the function: the
    // `;` below is the next item's, not this signature's.
    let err = fn_body("fn t<T() { let a = 1; }\nconst X: u8 = 1;\n", "t").unwrap_err();
    assert!(err.contains("`fn t`"), "{err}");
}

/// `fn  h(` and `fn\th(`: rustfmt never writes them, but a definition the scan
/// cannot find is a body it never reads.
#[test]
fn a_definition_is_found_through_any_whitespace_after_fn() {
    for gap in ["  ", "\t", " \t ", "\n"] {
        let src = format!("fn{gap}h() {{ let _rt = GpuRuntime::new(); }}\n#[test]\nfn t() {{ h(); }}\n");
        assert_eq!(
            fn_definitions(&blank_literals(&src), "h"),
            1,
            "{gap:?}: the definition was not found"
        );
        assert!(
            device_marker(&src, "t").unwrap().is_some(),
            "{gap:?}: the helper's device call went unseen"
        );
        // And as the test itself.
        let src = format!("#[test]\nfn{gap}t() {{ let _rt = GpuRuntime::new(); }}\n");
        assert!(device_marker(&src, "t").unwrap().is_some(), "{gap:?}: the test itself");
    }
    // A fn-pointer type and a longer name are not definitions of `h`.
    assert_eq!(fn_definitions("let f: fn(u8) -> u8; fn  hh() {}", "h"), 0);
}

/// The word boundary in the call walk. `my_helper(` is a call to `my_helper`, not
/// to `helper`: a callee whose name is the tail of another identifier must not
/// be followed, or a clean test is charged for an unrelated device helper.
#[test]
fn a_callee_that_is_the_tail_of_another_name_is_not_called() {
    let src = "fn helper() { let _rt = GpuRuntime::new(); }\n\
               fn my_helper() {}\n\
               #[test]\nfn t() { my_helper(); }\n";
    assert_eq!(
        device_marker(src, "t"),
        Ok(None),
        "`my_helper(` was read as a call to `helper`"
    );
    // The boundary is on the left only; a real call still counts around it.
    for call in [
        "helper();",
        "x.helper();",
        "Self::helper();",
        "(helper)();",
        "helper::<u8>();",
    ] {
        let src = format!("fn helper() {{ let _rt = GpuRuntime::new(); }}\n#[test]\nfn t() {{ {call} }}\n");
        let found = device_marker(&src, "t").unwrap();
        // `(helper)()` is a fn pointer call, which the scan does not follow.
        assert_eq!(found.is_some(), call != "(helper)();", "{call}");
    }
}

/// The remaining survivors of the adversary's round: `test_attribute_precedes`
/// reads the same definition sites as the scan, a `{` nested inside a
/// const-generic block stays inside it, and a `>` inside parentheses never closes
/// an angle bracket outside them.
#[test]
fn the_signature_scan_edges_the_adversary_found() {
    for gap in ["\t", "  ", " \t"] {
        let src = format!("#[test]\nfn{gap}t() {{}}\n");
        assert!(
            test_attribute_precedes(&src, "t"),
            "{gap:?}: the #[test] above `fn{gap}t` was not seen"
        );
    }
    assert!(!test_attribute_precedes("fn t() {}\n", "t"));
    let src = "fn h() where u8: Tr<{ { 1 }; 2 }> { let _rt = GpuRuntime::new(); }\n";
    assert_eq!(
        body_of(src, "h"),
        "{ let _rt = GpuRuntime::new(); }",
        "a nested block ended the const-generic block early"
    );
    let src = "fn h<F: Fn(Vec<u8>), const N: usize = { 3 }>() { let _rt = GpuRuntime::new(); }\n\
               #[test]\nfn t() { h(|_| ()); }\n";
    assert!(
        device_marker(src, "t")
            .unwrap()
            .is_some_and(|m| m.contains("h mentions GpuRuntime")),
        "a `>` inside parentheses closed the generics early"
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
