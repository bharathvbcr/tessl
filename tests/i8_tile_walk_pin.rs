//! Device-free pin: `kernels/i8_tile_walk.h` is the tile walk of
//! `kernels/gemm_coop.h`, under a stated and minimal normalisation.
//!
//! The int8 kernels cannot include `gemm_coop.h` (its `constexpr constant`
//! symbols are program-scope and duplicate at metallib link), so the walk is
//! copied with internal linkage. A copy that drifts mis-places tiles on the
//! int8 path only, and nothing in the build notices. This test extracts the
//! seven walk items from both headers and compares them token for token.
//!
//! Normalisation, applied to BOTH sides and nothing else:
//!   1. `//` and `/* */` comments are removed.
//!   2. Whitespace is not significant: the text is compared as tokens.
//!   3. The linkage keywords `static` and `constexpr` are dropped. Those are
//!      the whole difference between `inline` / `constexpr constant` in
//!      gemm_coop.h and `static inline` / `static constant` in i8_tile_walk.h.
//!   4. On the int8 side only, an identifier's leading `i8_` / `I8_` is
//!      removed (the documented rename that avoids a symbol clash). The coop
//!      side must carry no such prefix, so the rename cannot hide a difference.
//!
//!   5. Line splicing (backslash-newline) happens before comment removal, as in
//!      translation phase 2 of the C preprocessor, so a `//` comment ending in a
//!      backslash swallows the next line here exactly as it does in the compiler.
//!
//! Preprocessor directives outside the seven items could rename or retype what
//! they use (`#define i8_min max`), so every directive in the int8 header must
//! have an identical counterpart in gemm_coop.h after the same normalisation.
//! The one exemption is the include guard (`#ifndef G`, `#define G` with no
//! body, `G` used nowhere else, and the final `#endif`).
//!
//! Anything else (an operator, a literal, a type, a parameter, an item
//! present on one side only) fails.
//!
//! The comparison is deliberately conservative: it also refuses edits that
//! compute the same thing (a consistently renamed parameter, added
//! parentheses, `#pragma clang loop unroll(full)` for `#pragma unroll`). Those
//! stay refused so that no semantic judgement is made by this test; a
//! legitimate change is made to both headers, or to this test's rules. The negative self-tests below mutate a
//! copy of the text in memory, never the headers, and require the comparison
//! to reject every one-token change in either copy.

const GEMM_COOP_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gemm_coop.h"));
const I8_TILE_WALK_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/i8_tile_walk.h"));

/// One walk item: how to find it in each header (after comment removal).
/// `Fn` items run from the signature to the matching `}`; `Const` items run
/// from the declaration to `;`. `template` items must be preceded by
/// `template <int SM>`, which is part of what is compared.
struct Item {
    name: &'static str,
    coop: &'static str,
    i8: &'static str,
    kind: Kind,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Fn,
    TemplateFn,
    Const,
}

const ITEMS: [Item; 7] = [
    Item {
        name: "morton_decode_2d",
        coop: "uint2 morton_decode_2d(",
        i8: "uint2 i8_morton_decode_2d(",
        kind: Kind::Fn,
    },
    Item {
        name: "tile_grid_is_morton",
        coop: "bool tile_grid_is_morton(",
        i8: "bool i8_tile_grid_is_morton(",
        kind: Kind::Fn,
    },
    Item {
        name: "tile_from_linear",
        coop: "uint2 tile_from_linear(",
        i8: "uint2 i8_tile_from_linear(",
        kind: Kind::Fn,
    },
    Item {
        name: "tile_from_linear_panel",
        coop: "uint2 tile_from_linear_panel(",
        i8: "uint2 i8_tile_from_linear_panel(",
        kind: Kind::Fn,
    },
    Item {
        name: "PANEL_MIN_B_ELEMS",
        coop: "constexpr constant ulong PANEL_MIN_B_ELEMS",
        i8: "static constant ulong I8_PANEL_MIN_B_ELEMS",
        kind: Kind::Const,
    },
    Item {
        name: "PANEL_BAND_ROWS",
        coop: "constexpr constant uint PANEL_BAND_ROWS",
        i8: "static constant uint I8_PANEL_BAND_ROWS",
        kind: Kind::Const,
    },
    Item {
        name: "tile_walk",
        coop: "uint2 tile_walk(",
        i8: "uint2 i8_tile_walk(",
        kind: Kind::TemplateFn,
    },
];

/// Remove `//` and `/* */` comments, keeping newlines so nothing fuses.
fn strip_comments(src: &str) -> String {
    // Phase 2 before phase 3: a backslash-newline joins lines first, so a `//`
    // comment ending in a backslash continues onto the next line.
    let spliced = src.replace("\\\r\n", "").replace("\\\n", "");
    let src = spliced.as_str();
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            let end = src[i + 2..].find("*/").expect("unterminated block comment") + i + 4;
            out.push(' ');
            i = end;
        } else {
            let ch = src[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

const TWO_CHAR_OPS: [&str; 15] = [
    ">>", "<<", "&&", "||", "==", "!=", ">=", "<=", "|=", "&=", "+=", "-=", "++", "--", "::",
];

/// Identifiers and numbers (`1u`, `1ul`, `23`) are one token each, operators
/// of two characters are one token, every other non-space character is one.
fn tokens(src: &str) -> Vec<String> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_alphanumeric() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else {
            let two: String = chars[i..chars.len().min(i + 2)].iter().collect();
            if TWO_CHAR_OPS.contains(&two.as_str()) {
                out.push(two);
                i += 2;
            } else {
                out.push(c.to_string());
                i += 1;
            }
        }
    }
    out
}

/// Rule 3: the linkage keywords that differ between the two headers.
fn drop_linkage(toks: Vec<String>) -> Vec<String> {
    toks.into_iter().filter(|t| t != "static" && t != "constexpr").collect()
}

/// Rule 4: the int8 side's rename prefix. Refuses a coop token that carries
/// the prefix, so the strip cannot paper over a real difference.
fn strip_i8_prefix(toks: Vec<String>) -> Vec<String> {
    toks.into_iter()
        .map(|t| {
            t.strip_prefix("i8_")
                .or_else(|| t.strip_prefix("I8_"))
                .map(str::to_string)
                .unwrap_or(t)
        })
        .collect()
}

/// The text of one item in `src` (comments already removed): from `marker`
/// to the matching `}` (functions) or the `;` (constants). `marker` must occur
/// exactly once.
fn extract(src: &str, item: &str, marker: &str, kind: Kind) -> Result<String, String> {
    let at: Vec<usize> = src.match_indices(marker).map(|(i, _)| i).collect();
    if at.len() != 1 {
        return Err(format!("{item}: `{marker}` occurs {} times, expected 1", at.len()));
    }
    let start = at[0];
    let end = if kind == Kind::Const {
        start + src[start..].find(';').ok_or_else(|| format!("{item}: no `;`"))? + 1
    } else {
        let open = start + src[start..].find('{').ok_or_else(|| format!("{item}: no body"))?;
        let mut depth = 0usize;
        let mut end = None;
        for (i, ch) in src[open..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        end.ok_or_else(|| format!("{item}: unbalanced braces"))?
    };
    let mut text = src[start..end].to_string();
    if kind == Kind::TemplateFn {
        let before = tokens(&src[..start]);
        // The last tokens before the signature: `template <int SM> [static] inline`.
        let head: Vec<&str> = before.iter().rev().take(7).rev().map(String::as_str).collect();
        let t = head
            .iter()
            .position(|w| *w == "template")
            .ok_or_else(|| format!("{item}: not preceded by `template <int SM>`"))?;
        let decl = head[t..].join(" ");
        if !decl.starts_with("template < int SM >") {
            return Err(format!(
                "{item}: template head is `{decl}`, expected `template <int SM>`"
            ));
        }
        text = format!("template <int SM> {text}");
    }
    Ok(text)
}

/// One item with both sides normalised.
struct Pair {
    name: &'static str,
    coop: Vec<String>,
    i8: Vec<String>,
}

/// Both sides of every item, normalised.
fn normalised(coop_src: &str, i8_src: &str) -> Result<Vec<Pair>, String> {
    let coop_src = strip_comments(coop_src);
    let i8_src = strip_comments(i8_src);
    let mut out = Vec::new();
    for it in &ITEMS {
        let c = drop_linkage(tokens(&extract(&coop_src, it.name, it.coop, it.kind)?));
        if let Some(t) = c.iter().find(|t| t.starts_with("i8_") || t.starts_with("I8_")) {
            return Err(format!("{}: gemm_coop.h carries the int8 prefix in `{t}`", it.name));
        }
        let i = strip_i8_prefix(drop_linkage(tokens(&extract(&i8_src, it.name, it.i8, it.kind)?)));
        out.push(Pair {
            name: it.name,
            coop: c,
            i8: i,
        });
    }
    Ok(out)
}

/// Normalised text of each preprocessor directive of `src` (comments and
/// splices already removed), in order.
fn directives(src: &str) -> Vec<Vec<String>> {
    src.lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .map(tokens)
        .collect()
}

/// Refuse a directive of the int8 header that gemm_coop.h does not have,
/// naming it. The int8 include guard is exempt; the coop guard is dropped
/// from the reference set the same way.
fn check_directives(coop_src: &str, i8_src: &str) -> Result<(), String> {
    let coop = strip_comments(coop_src);
    let i8 = strip_comments(i8_src);
    let reference: Vec<String> = without_guard(directives(&coop), &coop)?
        .into_iter()
        .map(|d| drop_linkage(d).join(" "))
        .collect();
    for d in without_guard(directives(&i8), &i8)? {
        let text = strip_i8_prefix(drop_linkage(d.clone())).join(" ");
        if !reference.contains(&text) {
            return Err(format!(
                "directive `{}` in i8_tile_walk.h has no identical counterpart in gemm_coop.h",
                d.join(" ")
            ));
        }
    }
    Ok(())
}

/// `dirs` without its include guard: `#ifndef G`, `#define G` (no body, `G`
/// occurring exactly twice in the file) first, `#endif` last.
fn without_guard(mut dirs: Vec<Vec<String>>, src: &str) -> Result<Vec<Vec<String>>, String> {
    let guard = |d: &Vec<String>, kw: &str| d.len() == 3 && d[0] == "#" && d[1] == kw;
    if dirs.len() < 3 || !guard(&dirs[0], "ifndef") || !guard(&dirs[1], "define") || dirs[0][2] != dirs[1][2] {
        return Err("no include guard (`#ifndef G` / `#define G` first)".into());
    }
    let name = dirs[0][2].clone();
    if tokens(src).iter().filter(|t| **t == name).count() != 2 {
        return Err(format!("include guard `{name}` is used outside its two directives"));
    }
    if dirs.last().map(|d| d.join(" ")) != Some("# endif".to_string()) {
        return Err("the last directive is not the guard's `#endif`".into());
    }
    dirs.pop();
    dirs.drain(..2);
    Ok(dirs)
}

/// `Ok` when every item matches; otherwise the first difference, with the
/// item and the token index.
fn walks_match(coop_src: &str, i8_src: &str) -> Result<(), String> {
    check_directives(coop_src, i8_src)?;
    for Pair { name, coop: c, i8: i } in normalised(coop_src, i8_src)? {
        if c != i {
            let at = c
                .iter()
                .zip(&i)
                .position(|(a, b)| a != b)
                .unwrap_or(c.len().min(i.len()));
            return Err(format!(
                "{name}: differs at token {at}: gemm_coop.h `{}` against i8_tile_walk.h `{}` ({} against {} tokens)",
                c.get(at).map_or("<end>", String::as_str),
                i.get(at).map_or("<end>", String::as_str),
                c.len(),
                i.len()
            ));
        }
    }
    Ok(())
}

/// `src` with the one occurrence of `from` replaced by `to`; panics if `from`
/// is absent or ambiguous, so a mutation can never silently be a no-op.
fn mutate(src: &str, from: &str, to: &str) -> String {
    assert_eq!(
        src.matches(from).count(),
        1,
        "mutation anchor `{from}` must occur exactly once"
    );
    src.replacen(from, to, 1)
}

#[test]
fn the_int8_tile_walk_is_the_production_tile_walk() {
    walks_match(GEMM_COOP_H, I8_TILE_WALK_H).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn the_pin_extracts_every_item_with_a_body() {
    let n = normalised(GEMM_COOP_H, I8_TILE_WALK_H).unwrap();
    assert_eq!(n.len(), ITEMS.len());
    for Pair { name, coop: c, i8: i } in &n {
        assert!(
            c.len() >= 6,
            "{name}: only {} tokens extracted from gemm_coop.h",
            c.len()
        );
        assert_eq!(c.len(), i.len(), "{name}");
    }
    // The walk's decision logic is inside what was extracted.
    let walk = &n.iter().find(|p| p.name == "tile_walk").unwrap().coop;
    assert!(walk.join(" ").starts_with("template < int SM > uint2 tile_walk ("));
    assert!(walk.iter().any(|t| t == "PANEL_MIN_B_ELEMS"));
}

/// The normalisation lets the documented differences through and no others.
#[test]
fn comments_whitespace_and_linkage_are_not_differences() {
    let coop = format!("// leading\n{}\n/* tail */", GEMM_COOP_H.replace("    ", "\t"));
    let i8 = I8_TILE_WALK_H
        .replace("static inline", "static   inline")
        .replace('\n', "\n\n");
    walks_match(&coop, &i8).unwrap();
}

/// Every one-token change to either copy is rejected: for each token of each
/// item on each side, replace it with a different token and compare.
#[test]
fn every_one_token_change_to_either_copy_is_rejected() {
    let n = normalised(GEMM_COOP_H, I8_TILE_WALK_H).unwrap();
    let mut mutations = 0;
    for Pair { name, coop: c, i8: i } in &n {
        for side in 0..2 {
            let base = if side == 0 { c } else { i };
            for at in 0..base.len() {
                let mut m = base.clone();
                m[at] = format!("{}_mutated", m[at]);
                let (a, b) = if side == 0 { (&m, i) } else { (c, &m) };
                assert_ne!(a, b, "{name} side {side} token {at}: the mutation was a no-op");
                mutations += 1;
            }
        }
    }
    // The comparison itself is `Vec` equality; the loop above is the proof
    // that a changed token makes the two sides unequal. Pin the count so a
    // shrinking extraction (an item that stops matching) is visible.
    assert!(mutations > 300, "only {mutations} one-token mutations ran");
}

/// The same, end to end on the text: each mutation is a plausible edit of one
/// copy and must make `walks_match` fail, naming the item it broke.
#[test]
fn one_token_edits_to_the_text_of_either_header_are_rejected() {
    // (anchor in i8_tile_walk.h, replacement, item that must be named)
    let i8_edits: [(&str, &str, &str); 12] = [
        ("1ul << 23", "1ul << 24", "PANEL_MIN_B_ELEMS"),
        ("= 512u;", "= 256u;", "PANEL_BAND_ROWS"),
        (
            "&& !i8_tile_grid_is_morton(tiles_n, tiles_m)) {",
            "&& i8_tile_grid_is_morton(tiles_n, tiles_m)) {",
            "tile_walk",
        ),
        (">= I8_PANEL_MIN_B_ELEMS", "> I8_PANEL_MIN_B_ELEMS", "tile_walk"),
        (
            "I8_PANEL_BAND_ROWS / (uint)SM",
            "I8_PANEL_BAND_ROWS * (uint)SM",
            "tile_walk",
        ),
        (
            "uint local_h = min(ph,",
            "uint local_h = max(ph,",
            "tile_from_linear_panel",
        ),
        (
            "band * ph + rem % local_h",
            "band * ph + rem / local_h",
            "tile_from_linear_panel",
        ),
        (
            "}\n    return uint2(linear % tiles_n, linear / tiles_n);",
            "}\n    return uint2(linear / tiles_n, linear % tiles_n);",
            "tile_from_linear",
        ),
        ("tiles_n == tiles_m &&", "tiles_n <= tiles_m &&", "tile_grid_is_morton"),
        (
            "(tiles_n & (tiles_n - 1u)) == 0u",
            "(tiles_n & (tiles_n + 1u)) == 0u",
            "tile_grid_is_morton",
        ),
        ("(c >> (2 * i + 1)) & 1u", "(c >> (2 * i + 2)) & 1u", "morton_decode_2d"),
        ("i < 16; ++i", "i < 15; ++i", "morton_decode_2d"),
    ];
    for (from, to, item) in i8_edits {
        let err = walks_match(GEMM_COOP_H, &mutate(I8_TILE_WALK_H, from, to))
            .expect_err(&format!("`{from}` -> `{to}` in i8_tile_walk.h was accepted"));
        assert!(err.starts_with(item), "`{from}` -> `{to}`: wrong item named: {err}");
    }
    // The same edits made to the production header (its spelling of the anchor).
    for (from, to, item) in i8_edits {
        let (from, to) = (
            from.replace("i8_", "").replace("I8_", ""),
            to.replace("i8_", "").replace("I8_", ""),
        );
        let err = walks_match(&mutate(GEMM_COOP_H, &from, &to), I8_TILE_WALK_H)
            .expect_err(&format!("`{from}` -> `{to}` in gemm_coop.h was accepted"));
        assert!(
            err.starts_with(item),
            "`{from}` -> `{to}` (coop): wrong item named: {err}"
        );
    }
}

#[test]
fn a_missing_duplicated_or_prefixed_item_is_refused() {
    // A walk function gone from the int8 copy.
    let gone = mutate(
        I8_TILE_WALK_H,
        "uint2 i8_tile_from_linear(",
        "uint2 i8_tile_from_linear_x(",
    );
    assert!(walks_match(GEMM_COOP_H, &gone).unwrap_err().contains("occurs 0 times"));
    // A second definition of an item in the production header.
    let twice =
        format!("{GEMM_COOP_H}\ninline uint2 tile_from_linear(uint a, uint b, uint c) {{ return uint2(a, b); }}\n");
    assert!(walks_match(&twice, I8_TILE_WALK_H)
        .unwrap_err()
        .contains("occurs 2 times"));
    // The rename prefix may not leak into the production header.
    let leaked = mutate(
        GEMM_COOP_H,
        "return morton_decode_2d(linear);",
        "return i8_morton_decode_2d(linear);",
    );
    assert!(walks_match(&leaked, I8_TILE_WALK_H)
        .unwrap_err()
        .contains("carries the int8 prefix"));
    // The template head of `tile_walk` is compared too.
    let head = mutate(I8_TILE_WALK_H, "template <int SM>", "template <int SN>");
    assert!(walks_match(GEMM_COOP_H, &head).unwrap_err().starts_with("tile_walk"));
}

// ---------------------------------------------------------------------------
// Adversarial review (card ft-70f2141f): edits the first version of the pin
// accepted, and edits it refuses although they compute the same thing.
// ---------------------------------------------------------------------------

#[test]
fn a_define_outside_the_items_that_redefines_a_called_name_is_refused() {
    // `i8_min` strips to `min` and matches the item; the macro makes it `max`.
    let edited = mutate(I8_TILE_WALK_H, "uint local_h = min(ph,", "uint local_h = i8_min(ph,");
    let edited = mutate(
        &edited,
        "using namespace metal;",
        "using namespace metal;\n#define i8_min max",
    );
    let err = walks_match(GEMM_COOP_H, &edited).unwrap_err();
    assert!(err.contains("#") && err.contains("define i8_min max"), "{err}");
}

#[test]
fn a_define_that_retypes_a_parameter_is_refused() {
    // `i8_uint` strips to `uint`; the macro makes it `ushort`, which wraps `linear` at 65536.
    let edited = mutate(
        I8_TILE_WALK_H,
        "static inline uint2 i8_tile_from_linear(uint linear,",
        "static inline uint2 i8_tile_from_linear(i8_uint linear,",
    );
    let edited = mutate(
        &edited,
        "using namespace metal;",
        "using namespace metal;\n#define i8_uint ushort",
    );
    let err = walks_match(GEMM_COOP_H, &edited).unwrap_err();
    assert!(err.contains("define i8_uint ushort"), "{err}");
}

#[test]
fn a_conditional_or_stray_directive_is_refused() {
    let edited = mutate(
        I8_TILE_WALK_H,
        "using namespace metal;",
        "using namespace metal;\n#if 0\n#endif",
    );
    assert!(walks_match(GEMM_COOP_H, &edited).unwrap_err().contains("directive"));
    // A guard macro that is also used in the code is not a guard.
    let edited = mutate(
        I8_TILE_WALK_H,
        "uint x = 0, y = 0;",
        "uint x = TESSL_I8_TILE_WALK_H, y = 0;",
    );
    assert!(walks_match(GEMM_COOP_H, &edited).unwrap_err().contains("include guard"));
}

#[test]
fn a_directive_continued_with_a_backslash_is_one_directive() {
    let edited = mutate(
        I8_TILE_WALK_H,
        "using namespace metal;",
        "using namespace metal;\n#define i8_min \\\n    max",
    );
    assert!(walks_match(GEMM_COOP_H, &edited)
        .unwrap_err()
        .contains("define i8_min max"));
}

#[test]
fn a_backslash_continued_comment_hides_the_next_line_as_in_the_compiler() {
    // `clang -E` drops the `return i8_morton_decode_2d(linear);` line of this text.
    let edited = mutate(
        I8_TILE_WALK_H,
        "if (i8_tile_grid_is_morton(tiles_n, tiles_m)) {\n        return i8_morton_decode_2d(linear);",
        "if (i8_tile_grid_is_morton(tiles_n, tiles_m)) { // hidden \\\n        return i8_morton_decode_2d(linear);",
    );
    let err = walks_match(GEMM_COOP_H, &edited).unwrap_err();
    assert!(err.starts_with("tile_from_linear"), "{err}");
    // CRLF line endings splice the same way.
    let crlf = edited.replace('\n', "\r\n");
    assert!(walks_match(GEMM_COOP_H, &crlf).is_err());
}

/// Conservative refusals, documented in the file header: these compute the
/// same thing and are refused anyway.
#[test]
fn equivalent_rewrites_are_refused_conservatively() {
    let renamed = mutate(
        I8_TILE_WALK_H,
        "static inline uint2 i8_tile_from_linear_panel(uint linear, uint tiles_n, uint tiles_m, uint ph) {\n    if (linear >= tiles_n * tiles_m) return uint2(tiles_n, tiles_m);\n    uint band = linear / (ph * tiles_n);\n    uint rem = linear - band * ph * tiles_n;",
        "static inline uint2 i8_tile_from_linear_panel(uint lin, uint tiles_n, uint tiles_m, uint ph) {\n    if (lin >= tiles_n * tiles_m) return uint2(tiles_n, tiles_m);\n    uint band = lin / (ph * tiles_n);\n    uint rem = lin - band * ph * tiles_n;",
    );
    assert!(walks_match(GEMM_COOP_H, &renamed).is_err());
    let parens = mutate(I8_TILE_WALK_H, "= 1ul << 23;", "= (1ul << 23);");
    assert!(walks_match(GEMM_COOP_H, &parens).is_err());
    let pragma = mutate(I8_TILE_WALK_H, "#pragma unroll", "#pragma clang loop unroll(full)");
    assert!(walks_match(GEMM_COOP_H, &pragma).is_err());
}

// ---------------------------------------------------------------------------
// Second adversarial pass (card ft-c68232d9), written against the pin before
// e056730 and re-run after it. A `false_accept` test names an edit that changes
// what the int8 kernel computes while `walks_match` returns Ok; a
// `false_reject` test names an edit that computes the same thing and is refused.
// ---------------------------------------------------------------------------

#[test]
fn adv_false_accept_define_outside_the_items_redefines_min() {
    // `i8_min` is stripped to `min` and matches; the macro makes it `max`.
    let edited = mutate(I8_TILE_WALK_H, "uint local_h = min(ph,", "uint local_h = i8_min(ph,");
    let edited = mutate(
        &edited,
        "using namespace metal;",
        "using namespace metal;\n#define i8_min max",
    );
    assert!(
        walks_match(GEMM_COOP_H, &edited).is_err(),
        "walks_match accepted a header whose panel height is max(ph, ..) instead of min(ph, ..)"
    );
}

#[test]
fn adv_false_accept_define_outside_the_items_retypes_a_parameter() {
    // `i8_uint` strips to `uint`; the macro makes it `ushort`, which wraps `linear` at 65536.
    let edited = mutate(
        I8_TILE_WALK_H,
        "static inline uint2 i8_tile_from_linear(uint linear,",
        "static inline uint2 i8_tile_from_linear(i8_uint linear,",
    );
    let edited = mutate(
        &edited,
        "using namespace metal;",
        "using namespace metal;\n#define i8_uint ushort",
    );
    assert!(
        walks_match(GEMM_COOP_H, &edited).is_err(),
        "walks_match accepted a header whose `linear` is a ushort"
    );
}

#[test]
fn adv_false_accept_backslash_continued_comment_hides_the_morton_branch() {
    // `clang -E` on this text drops the `return i8_morton_decode_2d(linear);`
    // line (checked), so a Morton grid takes the row-major path. strip_comments
    // ends a `//` comment at the first newline and keeps the line as code.
    let edited = mutate(
        I8_TILE_WALK_H,
        "if (i8_tile_grid_is_morton(tiles_n, tiles_m)) {\n        return i8_morton_decode_2d(linear);",
        "if (i8_tile_grid_is_morton(tiles_n, tiles_m)) { // hidden \\\n        return i8_morton_decode_2d(linear);",
    );
    assert!(
        walks_match(GEMM_COOP_H, &edited).is_err(),
        "walks_match accepted a header whose Morton return is inside a comment"
    );
}

#[test]
#[ignore = "pin strictness: intended or defect? a consistent parameter rename on the int8 side is refused; equivalent_rewrites_are_refused_conservatively pins that refusal as documented; GitPulse card pending"]
fn adv_false_reject_renaming_a_parameter_consistently() {
    let edited = mutate(
        I8_TILE_WALK_H,
        "static inline uint2 i8_tile_from_linear_panel(uint linear, uint tiles_n, uint tiles_m, uint ph) {\n    if (linear >= tiles_n * tiles_m) return uint2(tiles_n, tiles_m);\n    uint band = linear / (ph * tiles_n);\n    uint rem = linear - band * ph * tiles_n;",
        "static inline uint2 i8_tile_from_linear_panel(uint lin, uint tiles_n, uint tiles_m, uint ph) {\n    if (lin >= tiles_n * tiles_m) return uint2(tiles_n, tiles_m);\n    uint band = lin / (ph * tiles_n);\n    uint rem = lin - band * ph * tiles_n;",
    );
    walks_match(GEMM_COOP_H, &edited).unwrap_or_else(|e| panic!("harmless rename refused: {e}"));
}

#[test]
#[ignore = "pin strictness: intended or defect? a parenthesised constant is refused; the same test pins that refusal as documented; GitPulse card pending"]
fn adv_false_reject_parenthesised_constant() {
    let edited = mutate(I8_TILE_WALK_H, "= 1ul << 23;", "= (1ul << 23);");
    walks_match(GEMM_COOP_H, &edited).unwrap_or_else(|e| panic!("harmless parentheses refused: {e}"));
}

#[test]
#[ignore = "pin strictness: intended or defect? an equivalent unroll pragma spelling is refused; the same test pins that refusal as documented; GitPulse card pending"]
fn adv_false_reject_equivalent_unroll_pragma() {
    let edited = mutate(I8_TILE_WALK_H, "#pragma unroll", "#pragma clang loop unroll(full)");
    walks_match(GEMM_COOP_H, &edited).unwrap_or_else(|e| panic!("equivalent pragma refused: {e}"));
}
