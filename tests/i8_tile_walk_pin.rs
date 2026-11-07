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
//! Anything else (an operator, a literal, a type, a parameter, an item
//! present on one side only) fails. The negative self-tests below mutate a
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

/// `Ok` when every item matches; otherwise the first difference, with the
/// item and the token index.
fn walks_match(coop_src: &str, i8_src: &str) -> Result<(), String> {
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
