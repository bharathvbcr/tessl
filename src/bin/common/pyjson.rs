//! Output text that matches what Python printed for the same data, and the one
//! statistic those scripts reported.
//!
//! The result files under `bench/results/` were written by `json.dumps(doc,
//! indent=2)` and by `%`-formatting, and the tools that replaced those scripts
//! keep the same bytes, so a regenerated artifact differs from a committed one
//! only where a number does. This is not a general JSON library: a document is
//! built as a [`Doc`], and rendering refuses a non-finite number rather than
//! writing the `NaN` / `Infinity` tokens Python would (which are not JSON).
//!
//! Included with `#[path]` by `gemm_parity`, `kernel_coverage`, `attn_tune` and
//! the integration tests of each; it has no dependency of its own.

#![allow(dead_code)] // Each includer uses a subset.

/// A JSON value in construction order. Objects keep the order they were built in,
/// as a Python dict does.
#[derive(Clone, Debug, PartialEq)]
pub enum Doc {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Doc>),
    Obj(Vec<(String, Doc)>),
}

impl Doc {
    pub fn str(s: impl Into<String>) -> Doc {
        Doc::Str(s.into())
    }

    pub fn uint(n: usize) -> Doc {
        Doc::Int(n as i64)
    }

    pub fn strs<S: AsRef<str>>(items: &[S]) -> Doc {
        Doc::Arr(items.iter().map(|s| Doc::str(s.as_ref())).collect())
    }

    pub fn floats(items: &[f64]) -> Doc {
        Doc::Arr(items.iter().map(|x| Doc::Float(*x)).collect())
    }

    /// `json.dumps(self, indent=2)`: two-space indent, `", "` never used, `": "`
    /// after a key, empty containers as `[]` / `{}`, non-ASCII escaped.
    pub fn dumps(&self) -> Result<String, String> {
        let mut out = String::new();
        self.write(&mut out, 0)?;
        Ok(out)
    }

    fn write(&self, out: &mut String, depth: usize) -> Result<(), String> {
        match self {
            Doc::Null => out.push_str("null"),
            Doc::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Doc::Int(i) => out.push_str(&i.to_string()),
            Doc::Float(x) if x.is_finite() => out.push_str(&float_repr(*x)),
            Doc::Float(x) => return Err(format!("refusing to write the non-finite number {x} into JSON")),
            Doc::Str(s) => escape(s, out),
            Doc::Arr(items) if items.is_empty() => out.push_str("[]"),
            Doc::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    out.push_str(&"  ".repeat(depth + 1));
                    item.write(out, depth + 1)?;
                }
                out.push('\n');
                out.push_str(&"  ".repeat(depth));
                out.push(']');
            }
            Doc::Obj(fields) if fields.is_empty() => out.push_str("{}"),
            Doc::Obj(fields) => {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    out.push_str(&"  ".repeat(depth + 1));
                    escape(k, out);
                    out.push_str(": ");
                    v.write(out, depth + 1)?;
                }
                out.push('\n');
                out.push_str(&"  ".repeat(depth));
                out.push('}');
            }
        }
        Ok(())
    }
}

/// `statistics.median` of a non-empty list: the middle value, or the mean of the
/// two middle ones.
pub fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// An object from `(key, value)` pairs, in the order given.
pub fn obj<const N: usize>(fields: [(&str, Doc); N]) -> Doc {
    Doc::Obj(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// `json.dumps`'s string form with `ensure_ascii=True`.
fn escape(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)` (and so `json.dumps`) for a finite number: the
/// shortest string that reads back as the same double, in positional form when
/// the decimal exponent is in `-4..16` and in `d.ddde+XX` form otherwise.
pub fn float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // `{:e}` is the shortest round-trip digit string: "-1.2345e-7".
    let sci = format!("{x:e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut s = String::new();
    if neg {
        s.push('-');
    }
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        if point <= 0 {
            s.push_str("0.");
            s.push_str(&"0".repeat((-point) as usize));
            s.push_str(&digits);
        } else if (point as usize) >= digits.len() {
            s.push_str(&digits);
            s.push_str(&"0".repeat(point as usize - digits.len()));
            s.push_str(".0");
        } else {
            s.push_str(&digits[..point as usize]);
            s.push('.');
            s.push_str(&digits[point as usize..]);
        }
    } else {
        s.push_str(&digits[..1]);
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push_str(&format!("e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()));
    }
    s
}

/// Python's `'%.{prec}e' % x` for a finite number: `1.23e-07`, `4.50e+00`.
pub fn fmt_e(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let sci = format!("{x:.prec$e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

/// Python's `str(float)` / `repr(float)`, including `inf` and `nan`.
pub fn float_str(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x < 0.0 { "-inf" } else { "inf" }.to_string()
    } else {
        float_repr(x)
    }
}

/// Python's `repr(str)` for the plain names (lanes, seeds, shapes) that appear
/// in messages: single-quoted unless the text holds a single quote.
pub fn quote(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{s}\"")
    } else {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

/// Python's `repr(list[str])`.
pub fn quote_list<S: AsRef<str>>(items: &[S]) -> String {
    format!(
        "[{}]",
        items.iter().map(|s| quote(s.as_ref())).collect::<Vec<_>>().join(", ")
    )
}
