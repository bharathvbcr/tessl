//! The text side of an SVG writer: XML escaping, number spelling and a width
//! estimate for laying text out without a font engine.
//!
//! Everything here is a pure function of its input, so the same figure always
//! comes out as the same bytes.

/// `s` as XML character data or an attribute value. The five predefined
/// entities are escaped; tabs and line breaks become a space (a figure's text
/// is laid out line by line, never by embedded breaks); characters XML 1.0
/// cannot carry at all (C0 controls, DEL, U+FFFE, U+FFFF) become U+FFFD so the
/// document stays well formed whatever the input held.
pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' | '\n' | '\r' => out.push(' '),
            c if (c as u32) < 0x20 || c == '\u{7f}' || c == '\u{fffe}' || c == '\u{ffff}' => out.push('\u{fffd}'),
            c => out.push(c),
        }
    }
    out
}

/// A coordinate or length, two decimals at most, no trailing zeros, never `-0`.
pub fn num(v: f64) -> String {
    let s = format!("{v:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() {
        "0".to_string()
    } else {
        s.to_string()
    }
}

/// Width in pixels of `s` set in the system sans stack at `size` px, from a
/// small per-character table (Helvetica metrics, rounded). An estimate, but a
/// deterministic one, and what every layout decision here is made from.
pub fn text_width(s: &str, size: f64) -> f64 {
    let em: f64 = s
        .chars()
        .map(|c| match c {
            'i' | 'l' | 'j' | '.' | ',' | ':' | ';' | '\'' | '|' | '!' | ' ' | 'I' | 't' | 'f' | '(' | ')' | '['
            | ']' => 0.29,
            'm' | 'w' | 'M' | 'W' | '@' | '%' => 0.84,
            'r' | '-' | '/' | '"' | '*' => 0.37,
            'A'..='Z' => 0.68,
            '0'..='9' => 0.56,
            c if c.is_ascii() => 0.54,
            _ => 1.0,
        })
        .sum();
    em * size
}

/// `s` broken into lines no wider than `max_w` at `size` px. Existing `\n`
/// breaks are kept; the rest breaks at spaces. A word wider than the line is
/// cut by character so no line can overflow.
pub fn wrap(s: &str, max_w: f64, size: f64) -> Vec<String> {
    let mut lines = Vec::new();
    for para in s.split('\n') {
        let mut cur = String::new();
        for word in para.split(' ').filter(|w| !w.is_empty()) {
            let mut word = word.to_string();
            while text_width(&word, size) > max_w && word.chars().count() > 1 {
                let mut cut = String::new();
                for c in word.chars() {
                    let mut probe = cut.clone();
                    probe.push(c);
                    if text_width(&probe, size) > max_w && !cut.is_empty() {
                        break;
                    }
                    cut.push(c);
                }
                if !cur.is_empty() {
                    lines.push(std::mem::take(&mut cur));
                }
                let rest: String = word.chars().skip(cut.chars().count()).collect();
                lines.push(cut);
                word = rest;
            }
            if cur.is_empty() {
                cur = word;
            } else if text_width(&format!("{cur} {word}"), size) <= max_w {
                cur.push(' ');
                cur.push_str(&word);
            } else {
                lines.push(std::mem::replace(&mut cur, word));
            }
        }
        lines.push(cur);
    }
    lines
}
