//! No kernel source spells an infinity in code.
//!
//! `build.rs` compiles every kernel with `-fmetal-math-mode=fast`. The IR that
//! produces marks every float compare `fast`, which includes `ninf`: the
//! compiler may assume no operand is infinite, so `m == -INFINITY` is a
//! comparison it is free to fold, and a running max seeded with `-INFINITY` is
//! a value it is free to assume away. Nothing measured has gone wrong, but
//! nothing obliges the GPU compiler to keep it that way.
//!
//! The safe forms are a finite identity (`-FLT_MAX`) with an explicit "has
//! seen a value" flag — `cross_entropy.metal`'s `first`, the attention
//! kernels' `l > 0` — or a seed taken from the data itself, as `reduce.metal`
//! and `qwen35_score.metal` do. This test keeps the infinities from coming
//! back: it strips comments and fails on any `INFINITY` token left.
//!
//! The same fast-math flags make three more spellings unsafe, and the lint
//! bans them too: `NAN`, `HUGE_VAL` and `MAXFLOAT` (constants an `fcmp fast`
//! treats as poison) and a self-compare (`v != v` is the constant `false`).
//! It does NOT ban `isfinite` / `isnan` / `isinf`: Metal defines them as
//! integer tests on the bit pattern (`(as_type<uint>(x) & 0x7fffffff) <op>
//! 0x7f800000`), which fast math cannot fold. `scripts/check_kernel_ir.sh`
//! checks the compiled IR for the same properties.

use std::path::{Path, PathBuf};

/// `src` with `//` and `/* */` comments replaced by spaces, newlines kept so
/// line numbers still match the file.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i < b.len() && !(b[i] == b'*' && b.get(i + 1) == Some(&b'/')) {
                out.push(if b[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
            i += 2;
        } else {
            let ch = src[i..].chars().next().expect("index is on a char boundary");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Every `.metal` and `.h` file under `kernels/`, recursively.
fn kernel_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("kernels directory is readable") {
            let path = entry.expect("readable directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "metal" || e == "h") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Lines of `code` holding `token` as a whole identifier.
fn token_lines(code: &str, token: &str) -> Vec<usize> {
    code.lines()
        .enumerate()
        .filter(|(_, line)| {
            line.match_indices(token).any(|(at, m)| {
                let before = line[..at].chars().next_back();
                let after = line[at + m.len()..].chars().next();
                !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
            })
        })
        .map(|(n, _)| n + 1)
        .collect()
}

fn infinity_lines(code: &str) -> Vec<usize> {
    token_lines(code, "INFINITY")
}

/// Identifiers that fast math makes poison or meaningless, besides `INFINITY`.
const POISON_TOKENS: [&str; 3] = ["NAN", "HUGE_VAL", "MAXFLOAT"];

/// The access path ending just before `end` (an `==`/`!=` operator, whitespace
/// already skipped): identifiers joined by `.`, `->` and `[..]` subscripts,
/// e.g. `a[i].x`. Empty when the text there is not such a path (a call
/// `f(x)`, a number, a closing paren).
fn operand_before(code: &[char], end: usize) -> String {
    let mut i = end;
    while i > 0 {
        let c = code[i - 1];
        if is_ident(c) || c == '.' {
            i -= 1;
        } else if c == '>' && i >= 2 && code[i - 2] == '-' {
            i -= 2;
        } else if c == ']' {
            let mut depth = 0usize;
            loop {
                if i == 0 {
                    return String::new();
                }
                i -= 1;
                match code[i] {
                    ']' => depth += 1,
                    '[' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        } else {
            break;
        }
    }
    normalize_path(&code[i..end])
}

/// The access path starting at `start` (first non-space after the operator).
/// Empty when it is not a path or is a call (`a == f(x)`).
fn operand_after(code: &[char], start: usize) -> String {
    let mut i = start;
    while i < code.len() {
        let c = code[i];
        if is_ident(c) || c == '.' {
            i += 1;
        } else if c == '-' && code.get(i + 1) == Some(&'>') {
            i += 2;
        } else if c == '[' {
            let mut depth = 0usize;
            loop {
                match code.get(i) {
                    None => return String::new(),
                    Some('[') => depth += 1,
                    Some(']') => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    Some(_) => {}
                }
                i += 1;
            }
        } else {
            break;
        }
    }
    let mut j = i;
    while code.get(j).is_some_and(|c| c.is_whitespace()) {
        j += 1;
    }
    if code.get(j) == Some(&'(') {
        return String::new();
    }
    normalize_path(&code[start..i])
}

/// `path` without whitespace, or empty when it does not start like an
/// identifier (a number, a bare `.`).
fn normalize_path(path: &[char]) -> String {
    let s: String = path.iter().filter(|c| !c.is_whitespace()).collect();
    if s.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') {
        s
    } else {
        String::new()
    }
}

/// Lines of `code` holding a self-compare: `==` or `!=` with the same access
/// path on both sides (`v != v`, `a[i] == a[i]`, `s.x != s.x`). Under fast
/// math the compiler may assume neither side is NaN, so the result is the
/// constant `true` / `false` and the NaN test it was meant to be is gone.
/// Calls are not paths and are not matched.
fn self_compare_lines(code: &str) -> Vec<usize> {
    let mut hits = Vec::new();
    for (n, line) in code.lines().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i + 1 < chars.len() {
            let is_op = chars[i + 1] == '='
                && (chars[i] == '=' || chars[i] == '!')
                // not `<==`, `>==`, `===`-style runs, nor `a <= b` / `a >= b`
                && !(i > 0 && matches!(chars[i - 1], '<' | '>' | '=' | '!'))
                && chars.get(i + 2) != Some(&'=');
            if is_op {
                let mut l = i;
                while l > 0 && chars[l - 1].is_whitespace() {
                    l -= 1;
                }
                let mut r = i + 2;
                while chars.get(r).is_some_and(|c| c.is_whitespace()) {
                    r += 1;
                }
                let left = operand_before(&chars, l);
                if !left.is_empty() && left == operand_after(&chars, r) {
                    hits.push(n + 1);
                    break;
                }
                i += 2;
            } else {
                i += 1;
            }
        }
    }
    hits
}

#[test]
fn the_scanner_finds_code_and_skips_comments() {
    let src = "a = -INFINITY; // INFINITY\n/* INFINITY\n INFINITY */ b = MY_INFINITY;\nc = INFINITY_X;\n#define X \\\n    (INFINITY)\n";
    assert_eq!(infinity_lines(&strip_comments(src)), vec![1, 6]);
}

#[test]
fn no_kernel_spells_an_infinity_in_code() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let sources = kernel_sources(&root);
    assert!(
        sources.len() > 20,
        "found only {} kernel sources under {}",
        sources.len(),
        root.display()
    );
    let mut hits = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("kernel source is readable UTF-8");
        for line in infinity_lines(&strip_comments(&src)) {
            hits.push(format!("{}:{line}", path.strip_prefix(&root).unwrap_or(path).display()));
        }
    }
    assert!(
        hits.is_empty(),
        "{} INFINITY use(s) in kernel code, which fast math may fold away; seed with -FLT_MAX \
         and a `seen` flag, or from the data, instead: {hits:#?}",
        hits.len()
    );
}

#[test]
fn the_poison_token_scanner_is_whole_identifier_and_skips_comments() {
    let src = "a = NAN; // NAN\n/* HUGE_VAL */ b = MAXFLOAT;\nc = MY_NAN + NAN_X + isnan(x);\nd = HUGE_VAL;\n";
    let code = strip_comments(src);
    assert_eq!(token_lines(&code, "NAN"), vec![1]);
    assert_eq!(token_lines(&code, "HUGE_VAL"), vec![4]);
    assert_eq!(token_lines(&code, "MAXFLOAT"), vec![2]);
}

#[test]
fn the_self_compare_matcher_finds_self_compares_and_nothing_else() {
    let hit = |src: &str| self_compare_lines(&strip_comments(src));
    // Self-compares: plain, member, indexed, arrow, spaced, both operators.
    assert_eq!(hit("if (v != v) x = 0;"), vec![1]);
    assert_eq!(hit("if (v == v) x = 0;"), vec![1]);
    assert_eq!(hit("ok = (a[i] != a[i]);"), vec![1]);
    assert_eq!(hit("ok = s.x != s.x;"), vec![1]);
    assert_eq!(hit("ok = p->v == p->v;"), vec![1]);
    assert_eq!(hit("ok = a[ i + 1 ]!=a[i+1];"), vec![1]);
    assert_eq!(hit("ok = m[r][c] != m[r][c];"), vec![1]);
    assert_eq!(hit("x = 1;\nif (acc != acc) {}\n"), vec![2]);
    // Not self-compares: different operands, a longer path on one side,
    // literals, calls, ordered compares, assignments, and comments.
    assert!(hit("if (v != w) {}").is_empty());
    assert!(hit("if (a[i] != a[j]) {}").is_empty());
    assert!(hit("if (a[i] != a[i + 1]) {}").is_empty());
    assert!(hit("if (p.a == a) {}").is_empty());
    assert!(hit("if (a == p.a) {}").is_empty());
    assert!(hit("if (a == a.x) {}").is_empty());
    assert!(hit("if (n == 0 || n != 1) {}").is_empty());
    assert!(hit("if (f(x) == f(x)) {}").is_empty());
    assert!(hit("if (v == v(1)) {}").is_empty());
    assert!(hit("if (v <= v || v >= v) {}").is_empty());
    assert!(hit("x = v; y = v;").is_empty());
    assert!(hit("// if (v != v) {}\n/* v == v */").is_empty());
    // isfinite / isnan / isinf are fast-math safe and never matched.
    assert!(hit("if (!isfinite(v) || isnan(v) || isinf(v)) {}").is_empty());
}

#[test]
fn no_kernel_uses_a_fast_math_poison_constant() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut hits = Vec::new();
    for path in kernel_sources(&root) {
        let src = std::fs::read_to_string(&path).expect("kernel source is readable UTF-8");
        let code = strip_comments(&src);
        for token in POISON_TOKENS {
            for line in token_lines(&code, token) {
                hits.push(format!(
                    "{}:{line} {token}",
                    path.strip_prefix(&root).unwrap_or(&path).display()
                ));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "{} use(s) of NAN / HUGE_VAL / MAXFLOAT in kernel code; under fast math a compare against \
         them is poison. Use a finite identity and a `seen` flag, or bit patterns: {hits:#?}",
        hits.len()
    );
}

#[test]
fn no_kernel_compares_a_value_with_itself() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut hits = Vec::new();
    for path in kernel_sources(&root) {
        let src = std::fs::read_to_string(&path).expect("kernel source is readable UTF-8");
        for line in self_compare_lines(&strip_comments(&src)) {
            hits.push(format!(
                "{}:{line}",
                path.strip_prefix(&root).unwrap_or(&path).display()
            ));
        }
    }
    assert!(
        hits.is_empty(),
        "{} self-compare(s) in kernel code; fast math folds `v != v` to false, so it is not a \
         NaN test. Use isnan / isfinite (integer bit tests in Metal): {hits:#?}",
        hits.len()
    );
}

/// `scripts/check_kernel_ir.sh` must compile with the flags `build.rs` does.
/// Both files spell each flag once; this fails when one of them stops.
#[test]
fn check_kernel_ir_uses_build_rs_flags() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let build = std::fs::read_to_string(root.join("build.rs")).expect("build.rs is readable");
    let script =
        std::fs::read_to_string(root.join("scripts/check_kernel_ir.sh")).expect("check_kernel_ir.sh is readable");
    for flag in [
        "-O2",
        "-fmetal-math-mode=fast",
        "-isysroot",
        "-mmacosx-version-min=26.0",
        "-ffp-contract=off",
        "qwen35_adamw.metal",
        "metal4.0",
        "metal3.2",
    ] {
        assert!(
            build.contains(flag),
            "build.rs no longer spells {flag}; update scripts/check_kernel_ir.sh"
        );
        assert!(
            script.contains(flag),
            "scripts/check_kernel_ir.sh does not spell {flag}, which build.rs uses"
        );
    }
}
