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

/// `src` with `//` and `/* */` comments replaced by spaces, and the contents of
/// string and char literals blanked (the quotes stay), newlines kept so line
/// numbers still match the file. A `//` inside a string does not open a
/// comment and `"a != a"` is not code.
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
        } else if b[i] == b'"' || (b[i] == b'\'' && !(i > 0 && b[i - 1].is_ascii_alphanumeric())) {
            // A `'` after an alphanumeric is a C++14 digit separator, not a char literal.
            let quote = b[i];
            out.push(quote as char);
            i += 1;
            while i < b.len() && b[i] != quote && b[i] != b'\n' {
                if b[i] == b'\\' && i + 1 < b.len() && b[i + 1] != b'\n' {
                    out.push(' ');
                    i += 1;
                }
                out.push(' ');
                i += 1;
            }
            if i < b.len() && b[i] == quote {
                out.push(quote as char);
                i += 1;
            }
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

/// `code` as a char vector with each line number, whitespace dropped except a
/// single space between two identifier characters (`return v` keeps its
/// space; `v\n    != v` becomes `v!=v`). Expressions split over lines then
/// read as one.
fn collapse(code: &str) -> (Vec<char>, Vec<usize>) {
    let (mut text, mut lines) = (Vec::new(), Vec::new());
    let (mut line, mut gap) = (1usize, false);
    for c in code.chars() {
        if c.is_whitespace() {
            gap = true;
            if c == '\n' {
                line += 1;
            }
            continue;
        }
        if gap && text.last().is_some_and(|&p| is_ident(p)) && is_ident(c) {
            text.push(' ');
            lines.push(line);
        }
        gap = false;
        text.push(c);
        lines.push(line);
    }
    (text, lines)
}

/// Index of the bracket closing the one at `open`, counting nesting of that
/// bracket kind only.
fn matching_close(t: &[char], open: usize) -> Option<usize> {
    let (o, c) = match t[open] {
        '(' => ('(', ')'),
        '[' => ('[', ']'),
        '<' => ('<', '>'),
        _ => return None,
    };
    let mut depth = 0usize;
    for (k, &ch) in t.iter().enumerate().skip(open) {
        if ch == o {
            depth += 1;
        } else if ch == c {
            depth -= 1;
            if depth == 0 {
                return Some(k);
            }
        }
    }
    None
}

/// Index of the bracket opening the one that closes at `close`.
fn matching_open(t: &[char], close: usize) -> Option<usize> {
    let (o, c) = match t[close] {
        ')' => ('(', ')'),
        ']' => ('[', ']'),
        _ => return None,
    };
    let mut depth = 0usize;
    for k in (0..=close).rev() {
        if t[k] == c {
            depth += 1;
        } else if t[k] == o {
            depth -= 1;
            if depth == 0 {
                return Some(k);
            }
        }
    }
    None
}

/// Positions of `<` and `>` that bracket template arguments (`as_type<uint>(`,
/// `static_cast<float>(`, `vec<float, 4>`), so they are not read as compares.
/// A `<` right after an identifier whose matching `>` encloses only type-ish
/// text and is followed by `(`, `::`, `>` or `,`.
fn template_brackets(t: &[char]) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    for i in 1..t.len() {
        if t[i] != '<' || !is_ident(t[i - 1]) {
            continue;
        }
        let Some(j) = matching_close(t, i) else { continue };
        let inner_ok = t[i + 1..j]
            .iter()
            .all(|&c| is_ident(c) || matches!(c, ' ' | ',' | ':' | '<' | '>' | '*' | '&'));
        let after_ok = t.get(j + 1).is_some_and(|&c| matches!(c, '(' | ':' | '>' | ','));
        if inner_ok && after_ok {
            set.insert(i);
            set.insert(j);
        }
    }
    set
}

/// Types a cast to is transparent for a self-compare.
const CAST_TYPES: [&str; 8] = ["float", "half", "bfloat", "double", "int", "uint", "short", "ushort"];
const CAST_TEMPLATES: [&str; 4] = ["static_cast", "reinterpret_cast", "const_cast", "as_type"];

/// `s` without one transparent layer: outer parentheses, a C-style `(float)`
/// or functional `float(..)` cast, `static_cast<T>(..)` / `as_type<T>(..)`, or
/// a unary `*` / `&`.
fn peel(s: &str) -> Option<String> {
    let t: Vec<char> = s.chars().collect();
    if t.is_empty() {
        return None;
    }
    if t[0] == '*' || t[0] == '&' {
        return Some(t[1..].iter().collect());
    }
    if t[0] == '(' {
        let k = matching_close(&t, 0)?;
        if k == t.len() - 1 {
            return Some(t[1..k].iter().collect());
        }
        let inner: String = t[1..k].iter().collect();
        if CAST_TYPES.contains(&inner.as_str()) {
            return Some(t[k + 1..].iter().collect());
        }
        return None;
    }
    let name: String = t.iter().take_while(|&&c| is_ident(c)).collect();
    let n = name.len();
    if CAST_TYPES.contains(&name.as_str()) && t.get(n) == Some(&'(') && matching_close(&t, n) == Some(t.len() - 1) {
        return Some(t[n + 1..t.len() - 1].iter().collect());
    }
    if CAST_TEMPLATES.contains(&name.as_str()) && t.get(n) == Some(&'<') {
        let j = matching_close(&t, n)?;
        if t.get(j + 1) == Some(&'(') && matching_close(&t, j + 1) == Some(t.len() - 1) {
            return Some(t[j + 2..t.len() - 1].iter().collect());
        }
    }
    None
}

/// The operand `s` with every transparent layer peeled, or empty when what is
/// left is not an access path (a number, a call, an expression).
fn normalize_operand(s: &str) -> String {
    let mut cur = s.to_string();
    while let Some(next) = peel(&cur) {
        cur = next;
    }
    let mut depth = 0usize;
    for c in cur.chars() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            '(' if depth == 0 => return String::new(),
            _ => {}
        }
    }
    if cur.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') {
        cur
    } else {
        String::new()
    }
}

/// The operand ending just before `end`: identifiers joined by `.`, `::`,
/// `->`, with `[..]` subscripts, `(..)` groups and `<..>` template arguments.
fn left_operand(t: &[char], end: usize, tmpl: &std::collections::HashSet<usize>) -> String {
    let mut i = end;
    while i > 0 {
        let c = t[i - 1];
        if is_ident(c) || c == '.' {
            i -= 1;
        } else if i >= 2 && ((c == ':' && t[i - 2] == ':') || (c == '>' && t[i - 2] == '-')) {
            i -= 2;
        } else if c == ']' || c == ')' {
            match matching_open(t, i - 1) {
                Some(o) => i = o,
                None => return String::new(),
            }
        } else if c == '>' && tmpl.contains(&(i - 1)) {
            let Some(o) = (0..i - 1)
                .rev()
                .find(|k| tmpl.contains(k) && t[*k] == '<' && matching_close(t, *k) == Some(i - 1))
            else {
                return String::new();
            };
            i = o;
        } else {
            break;
        }
    }
    normalize_operand(&t[i..end].iter().collect::<String>())
}

/// The operand starting at `start`, as [`left_operand`] but forwards, after
/// any unary `*` / `&`.
fn right_operand(t: &[char], start: usize) -> String {
    let mut i = start;
    while t.get(i).is_some_and(|&c| c == '*' || c == '&') {
        i += 1;
    }
    let from = start;
    while i < t.len() {
        let c = t[i];
        if is_ident(c) || c == '.' {
            i += 1;
        } else if (c == ':' && t.get(i + 1) == Some(&':')) || (c == '-' && t.get(i + 1) == Some(&'>')) {
            i += 2;
        } else if c == '[' || c == '(' || (c == '<' && i > from && is_ident(t[i - 1])) {
            match matching_close(t, i) {
                Some(k) => i = k + 1,
                None => break,
            }
        } else {
            break;
        }
    }
    normalize_operand(&t[from..i].iter().collect::<String>())
}

/// Lines of `code` holding a self-compare: `==`, `!=`, `<`, `<=`, `>` or `>=`
/// with the same operand on both sides (`v != v`, `!(v < v)`, `a[i] == a[i]`,
/// `((x) != (x))`, `float(x) != float(x)`, `*p != *p`, `ns::v != ns::v`),
/// found over comment-stripped text with whitespace collapsed, so an operand
/// or the operator on another line is still seen. Under fast math the compiler
/// may assume neither side is NaN, so the result is a constant and the NaN
/// test it was meant to be is gone. Calls and expressions are not operands.
fn self_compare_lines(code: &str) -> Vec<usize> {
    let (t, lines) = collapse(code);
    let tmpl = template_brackets(&t);
    let mut hits = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let (c, next) = (t[i], t.get(i + 1).copied());
        let prev = if i > 0 { Some(t[i - 1]) } else { None };
        let len = match (c, next) {
            ('=' | '!', Some('=')) if !matches!(prev, Some('<' | '>' | '=' | '!')) && t.get(i + 2) != Some(&'=') => 2,
            ('<' | '>', Some('=')) if !matches!(prev, Some('<' | '>' | '-')) && t.get(i + 2) != Some(&'=') => 2,
            ('<' | '>', n)
                if !tmpl.contains(&i)
                    && !matches!(prev, Some('<' | '>' | '-'))
                    && !matches!(n, Some('<' | '>' | '=')) =>
            {
                1
            }
            _ => 0,
        };
        if len == 0 {
            i += 1;
            continue;
        }
        let left = left_operand(&t, i, &tmpl);
        if !left.is_empty() && left == right_operand(&t, i + len) {
            hits.push(lines[i]);
        }
        i += len;
    }
    hits.dedup();
    hits
}

/// Hex literal `0x...` (optional `u`/`l` suffix) at the start of `s`, as its
/// value.
fn hex_literal(s: &str) -> Option<u64> {
    let digits = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    let end = digits.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(digits.len());
    let rest = &digits[end..];
    if end == 0 || rest.starts_with(|c: char| is_ident(c) && !matches!(c, 'u' | 'U' | 'l' | 'L')) {
        return None;
    }
    u64::from_str_radix(&digits[..end], 16).ok()
}

/// Lines of `code` spelling an infinity or NaN other than the plain
/// identifiers: `isunordered`, `__builtin_inf*` / `__builtin_nan*`,
/// `numeric_limits<..>::infinity` / `quiet_NaN` / `signaling_NaN`, and an
/// `as_type<float|half|bfloat>(0x..)` whose literal has an all-ones exponent
/// for its width. Each hit is `(line, what)`.
fn inf_nan_spelling_lines(code: &str) -> Vec<(usize, String)> {
    let mut hits = Vec::new();
    for (n, line) in code.lines().enumerate() {
        let n = n + 1;
        let mut words = Vec::new();
        let mut start = None;
        for (k, c) in line.char_indices().chain(std::iter::once((line.len(), ' '))) {
            match (is_ident(c), start) {
                (true, None) => start = Some(k),
                (false, Some(s)) => {
                    words.push((s, &line[s..k]));
                    start = None;
                }
                _ => {}
            }
        }
        for &(at, w) in &words {
            let after_scope = line[..at].trim_end().ends_with("::");
            if w == "isunordered"
                || w.starts_with("__builtin_inf")
                || w.starts_with("__builtin_nan")
                || (after_scope && matches!(w, "infinity" | "quiet_NaN" | "signaling_NaN"))
            {
                hits.push((n, w.to_string()));
            }
            if w == "as_type" {
                let rest = &line[at + w.len()..];
                let Some(rest) = rest.trim_start().strip_prefix('<') else {
                    continue;
                };
                let Some(close) = rest.find('>') else { continue };
                let exp_mask: u64 = match rest[..close].trim() {
                    "float" => 0x7f80_0000,
                    "half" => 0x7c00,
                    "bfloat" => 0x7f80,
                    _ => continue,
                };
                let Some(arg) = rest[close + 1..].trim_start().strip_prefix('(') else {
                    continue;
                };
                if hex_literal(arg.trim_start()).is_some_and(|v| v & exp_mask == exp_mask) {
                    hits.push((n, format!("as_type<{}>(inf/NaN bits)", rest[..close].trim())));
                }
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
    assert_eq!(hit("if (v <= v || v >= v) {}"), vec![1]);
    assert_eq!(hit("if (!(v < v)) {}"), vec![1]);
    assert!(hit("if (v < w || w > v) {}").is_empty());
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
        let name = path.strip_prefix(&root).unwrap_or(&path).display().to_string();
        for token in POISON_TOKENS {
            for line in token_lines(&code, token) {
                hits.push(format!("{name}:{line} {token}"));
            }
        }
        for (line, what) in inf_nan_spelling_lines(&code) {
            hits.push(format!("{name}:{line} {what}"));
        }
    }
    assert!(
        hits.is_empty(),
        "{} use(s) of NAN / HUGE_VAL / MAXFLOAT / isunordered / __builtin_inf|nan / numeric_limits inf|NaN / inf-NaN as_type bit casts in kernel code; under fast math a compare against \
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

/// Lines of `code` that compare a value with `-FLT_MAX`: `x > -FLT_MAX`,
/// `-FLT_MAX < x`, `==`, `!=`, `<=`, `>=`, over collapsed text. `-FLT_MAX` is
/// the finite identity a running max is seeded with; compared against, it is a
/// stand-in for `-inf` (`x > -FLT_MAX` as "x is finite"), and under fast math
/// (`ninf`, `nnan`) that compare may be folded, and is false for NaN, which it
/// then silently drops. Classify infinities and NaN by bit pattern
/// (`as_type<uint>(x) == 0xff800000u`) or with `isfinite` / `isnan` / `isinf`.
/// An assignment (`m = -FLT_MAX`) and a `+FLT_MAX` sentinel are not hits.
fn neg_flt_max_compare_lines(code: &str) -> Vec<usize> {
    let (t, lines) = collapse(code);
    let needle: Vec<char> = "-FLT_MAX".chars().collect();
    let is_cmp = |a: char, b: Option<char>| matches!((a, b), ('=' | '!' | '<' | '>', Some('=')) | ('<' | '>', _));
    let mut hits = Vec::new();
    let mut i = 0;
    while i + needle.len() <= t.len() {
        if t[i..i + needle.len()] != needle[..] || (i > 0 && is_ident(t[i - 1])) {
            i += 1;
            continue;
        }
        let j = i + needle.len();
        if t.get(j).is_some_and(|&c| is_ident(c)) {
            i += 1;
            continue;
        }
        // Before: `<`, `>`, `<=`, `>=`, `==`, `!=` (not `->`, `<<`, `>>`, a lone `=`).
        let before = match (i.checked_sub(2).map(|k| t[k]), i.checked_sub(1).map(|k| t[k])) {
            (_, Some('=')) => matches!(i.checked_sub(2).map(|k| t[k]), Some('=' | '!' | '<' | '>')),
            (p, Some(c @ ('<' | '>'))) => p != Some('-') && p != Some(c),
            _ => false,
        };
        let after = match (t.get(j).copied(), t.get(j + 1).copied()) {
            (Some(a), b) if is_cmp(a, b) => !(matches!(a, '<' | '>') && b == Some(a)) && !(a == '=' && b != Some('=')),
            _ => false,
        };
        if before || after {
            hits.push(lines[i]);
        }
        i = j;
    }
    hits.dedup();
    hits
}

#[test]
fn the_neg_flt_max_compare_matcher_finds_compares_and_not_seeds() {
    let hit = |src: &str| neg_flt_max_compare_lines(&strip_comments(src));
    for src in [
        "if (x > -FLT_MAX) {}",
        "if (x >= -FLT_MAX) {}",
        "if (x < -FLT_MAX) {}",
        "if (x <= -FLT_MAX) {}",
        "if (x == -FLT_MAX) {}",
        "if (x != -FLT_MAX) {}",
        "if (-FLT_MAX < x) {}",
        "if (-FLT_MAX == x) {}",
        "g = z > -FLT_MAX ? a : b;",
        "if (x\n  > -FLT_MAX) {}",
    ] {
        assert_eq!(hit(src).len(), 1, "{src}");
    }
    for src in [
        "float m = -FLT_MAX;",
        "m = -FLT_MAX;",
        "partials[i] = -FLT_MAX;",
        "float v = max(v, -FLT_MAX);",
        "x = a < b ? -FLT_MAX : b;",
        "if (r < FLT_MAX) {}",
        "// if (x > -FLT_MAX) {}",
        "y = x << 1; z = -FLT_MAX_ISH;",
    ] {
        assert!(hit(src).is_empty(), "{src}");
    }
}

#[test]
fn no_kernel_compares_a_value_with_neg_flt_max() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut hits = Vec::new();
    for path in kernel_sources(&root) {
        let src = std::fs::read_to_string(&path).expect("kernel source is readable UTF-8");
        for line in neg_flt_max_compare_lines(&strip_comments(&src)) {
            hits.push(format!(
                "{}:{line}",
                path.strip_prefix(&root).unwrap_or(&path).display()
            ));
        }
    }
    assert!(
        hits.is_empty(),
        "{} compare(s) with -FLT_MAX in kernel code; as a stand-in for -inf it is a float compare \
         fast math may fold, and it drops NaN. Test the bits (as_type<uint>(x) == 0xff800000u) or \
         use isfinite / isinf: {hits:#?}",
        hits.len()
    );
}

// ---------------------------------------------------------------------------
// Adversarial regressions: spellings a kernel can use that a line-by-line,
// path-only self-compare scan missed.
// ---------------------------------------------------------------------------

#[test]
fn self_compare_of_parenthesised_and_cast_operands() {
    let hit = |src: &str| self_compare_lines(&strip_comments(src));
    assert_eq!(hit("#define ISN(x) ((x) != (x))"), vec![1], "parenthesised operands");
    assert_eq!(hit("auto b = (x) == (x);"), vec![1], "parenthesised operands");
    assert_eq!(hit("if (float(x) != float(x)) {}"), vec![1], "functional cast");
    assert_eq!(
        hit("if (static_cast<float>(x) != static_cast<float>(x)) {}"),
        vec![1],
        "static_cast"
    );
    assert_eq!(hit("if (*p != *p) {}"), vec![1], "dereference");
    assert_eq!(hit("if (ns::v != ns::v) {}"), vec![1], "scope-qualified name");
}

#[test]
fn self_compare_split_across_lines() {
    let hit = |src: &str| self_compare_lines(&strip_comments(src));
    assert_eq!(hit("if (v\n    != v) {}").len(), 1, "operator on the next line");
    assert_eq!(hit("if (v !=\n    v) {}").len(), 1, "operand on the next line");
}

#[test]
fn slash_slash_inside_a_string_does_not_hide_the_rest_of_the_line() {
    let hit = |src: &str| self_compare_lines(&strip_comments(src));
    assert_eq!(hit("const char *s = \"//\"; if (v != v) {}"), vec![1]);
}

/// The `-` flag literals of `fn metal_args` in `build.rs`.
fn metal_args_flags(build_rs: &str) -> Vec<String> {
    let start = build_rs.find("fn metal_args").expect("build.rs defines metal_args");
    let body = &build_rs[start..];
    let end = body.find("\n}\n").expect("metal_args ends");
    let mut flags = Vec::new();
    for lit in body[..end].split('"').skip(1).step_by(2) {
        // `-std={metal_std}` is the dialect, spelled per source by the script.
        if lit.starts_with('-') && !lit.starts_with("-std=") {
            flags.push(lit.to_string());
        }
    }
    flags
}

#[test]
fn the_metal_args_extractor_sees_a_planted_flag() {
    let planted = "fn metal_args() -> Vec<String> {\n    let mut args = [\"-O2\", \"-std={x}\"];\n    if cond {\n        args.push(\"-fno-fast-math\".into());\n    }\n    args\n}\n";
    assert_eq!(metal_args_flags(planted), vec!["-O2", "-fno-fast-math"]);
}

/// Every flag literal in `metal_args` is either spelled by the script or on the
/// explicit list of flags that cannot change the IR (`-Wall -Werror` only
/// turn warnings into errors under TESSL_METAL_WERROR). Derived from build.rs,
/// not hand-listed, so a flag added there is noticed. The sources that get
/// `-ffp-contract=off` (`disable_fp_contract`) and both `-std` dialects must be
/// spelled by the script too.
#[test]
fn every_build_rs_flag_is_in_the_ir_script_or_exempt() {
    const IR_NEUTRAL: [&str; 2] = ["-Wall", "-Werror"];
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let build = std::fs::read_to_string(root.join("build.rs")).expect("build.rs is readable");
    let script =
        std::fs::read_to_string(root.join("scripts/check_kernel_ir.sh")).expect("check_kernel_ir.sh is readable");
    let flags = metal_args_flags(&build);
    assert!(flags.len() >= 5, "extractor found only {flags:?}");
    for flag in flags {
        assert!(
            IR_NEUTRAL.contains(&flag.as_str()) || script.contains(&flag),
            "build.rs passes {flag} to metal but scripts/check_kernel_ir.sh does not"
        );
    }
    let off = &build[build
        .find("fn disable_fp_contract")
        .expect("build.rs defines disable_fp_contract")..];
    let off = &off[..off.find("\n}\n").expect("disable_fp_contract ends")];
    let off_sources: Vec<&str> = off
        .split('"')
        .skip(1)
        .step_by(2)
        .filter(|l| l.ends_with(".metal"))
        .collect();
    assert!(!off_sources.is_empty(), "no -ffp-contract=off source found in build.rs");
    for src in off_sources {
        assert!(
            script.contains(src),
            "build.rs disables fp-contract for {src}; the IR script does not"
        );
    }
    for std in ["metal4.0", "metal3.2"] {
        assert!(
            build.contains(std) && script.contains(std),
            "-std={std} not spelled by both"
        );
    }
}

#[test]
fn self_compare_matcher_negatives_and_every_operator() {
    let hit = |src: &str| self_compare_lines(&strip_comments(src));
    // Operands that merely end or start alike are different operands.
    for src in [
        "x1 != x",
        "ax != x",
        "x != x1",
        "x != ax",
        "a != b",
        "x.x != x",
        "a[i] < a[j]",
    ] {
        assert!(hit(&format!("if ({src}) {{}}")).is_empty(), "{src}");
    }
    // Every relational operator, bare or under `!(..)`, parens or casts.
    for op in ["==", "!=", "<", "<=", ">", ">="] {
        for src in [
            format!("if (v {op} v) {{}}"),
            format!("if (!(v {op} v)) {{}}"),
            format!("b = ((v) {op} (v));"),
            format!("b = float(v) {op} (float)v;"),
            format!("b = as_type<float>(v) {op} as_type<float>(v);"),
            format!("b = *p {op} *p;"),
        ] {
            assert_eq!(hit(&src), vec![1], "{src}");
        }
    }
    // Template brackets and arrows are not compares.
    assert!(hit("auto b = as_type<uint>(x) & 0x7f800000u; y = p->v;").is_empty());
    // Text in a string or char literal is not code, a digit separator is not a literal.
    assert!(hit("const char *s = \"a != a\"; char c = '!'; int n = 1'000;").is_empty());
    assert_eq!(hit("char c = '\\''; if (v != v) {}"), vec![1]);
    // Reported at the operator's line.
    assert_eq!(hit("if (v\n  !=\n  v) {}"), vec![2]);
}

#[test]
fn inf_nan_spellings_are_found_and_plain_bit_tests_are_not() {
    let hit = |src: &str| -> Vec<String> {
        inf_nan_spelling_lines(&strip_comments(src))
            .into_iter()
            .map(|(_, w)| w)
            .collect()
    };
    assert_eq!(hit("b = isunordered(a, c);"), vec!["isunordered"]);
    assert_eq!(
        hit("x = __builtin_inff(); y = __builtin_nanf(\"\");"),
        vec!["__builtin_inff", "__builtin_nanf"]
    );
    assert_eq!(
        hit("a = numeric_limits<float>::infinity(); b = std::numeric_limits<half>::quiet_NaN(); c = numeric_limits<float>::signaling_NaN();"),
        vec!["infinity", "quiet_NaN", "signaling_NaN"]
    );
    assert_eq!(hit("x = as_type<float>(0x7f800000u);").len(), 1);
    assert_eq!(hit("x = as_type<float>(0xFFC00000);").len(), 1);
    assert_eq!(hit("x = as_type<float>(0x7fa00000u);").len(), 1);
    assert_eq!(hit("x = as_type<half>(0x7C00);").len(), 1);
    assert_eq!(hit("x = as_type<half>(0xfe00);").len(), 1);
    assert_eq!(hit("x = as_type<bfloat>(0x7f80);").len(), 1);
    assert_eq!(
        hit("x = as_type<bfloat>(ushort(0x7fc0));").len(),
        0,
        "only a bare literal is matched"
    );
    // Finite bit patterns, other target types, non-literals, comments and the
    // fast-math-safe classification calls are fine.
    for ok in [
        "x = as_type<float>(0x3f800000u);",
        "x = as_type<half>(0x3c00);",
        "x = as_type<bfloat>(0x3f80);",
        "x = as_type<uint>(0x7f800000u);",
        "x = as_type<float>(bits);",
        "// as_type<float>(0x7f800000u) isunordered",
        "if (isfinite(x) || isnan(x) || isinf(x)) {}",
        "const uint infinity = 1; int quiet_NaN = 2;",
    ] {
        assert!(hit(ok).is_empty(), "{ok}");
    }
}
