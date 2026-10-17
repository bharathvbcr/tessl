//! The slice of Python these report scripts relied on, reproduced exactly so
//! the regenerated `.txt` and `.json` files are byte-identical to the ones the
//! scripts wrote: `int`/`float` that stay distinct, `sum` (CPython 3.12+
//! Neumaier), `statistics.median`, `min`/`max` returning the element,
//! `float.__repr__`, `format(x, ".2f")` and `".1%"`, `str.strip`,
//! `str.splitlines`, `Path.read_text`, and `json.dumps` (both the
//! `indent=1` and the `separators=(",", ":")` forms, `ensure_ascii` on).
//!
//! Where Python would raise (a missing key, a division by zero, `min` of an
//! empty list), these return `Err` and the tool exits 1 with the message, so
//! a malformed input stops the run rather than writing a wrong table.

use crate::json::{Json, NumForm};
use std::cmp::Ordering;
use std::path::Path;

pub type R<T> = Result<T, String>;

// ---------------------------------------------------------------------------
// Numbers: a JSON number keeps the type it was written with.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum Num {
    I(i128),
    F(f64),
}

pub use Num::{F, I};

impl Num {
    pub fn f(self) -> f64 {
        match self {
            I(i) => i as f64,
            F(x) => x,
        }
    }

    /// Python truthiness.
    pub fn truthy(self) -> bool {
        match self {
            I(i) => i != 0,
            F(x) => x != 0.0,
        }
    }

    /// `a < b`, `a == b`, `a > b` between ints and floats, exact. `None` when a NaN is involved.
    pub fn cmp(self, other: Num) -> Option<Ordering> {
        match (self, other) {
            (I(a), I(b)) => Some(a.cmp(&b)),
            (F(a), F(b)) => a.partial_cmp(&b),
            (I(a), F(b)) => int_vs_float(a, b),
            (F(a), I(b)) => int_vs_float(b, a).map(Ordering::reverse),
        }
    }

    pub fn lt(self, other: Num) -> bool {
        self.cmp(other) == Some(Ordering::Less)
    }

    pub fn gt(self, other: Num) -> bool {
        self.cmp(other) == Some(Ordering::Greater)
    }

    pub fn eq(self, other: Num) -> bool {
        self.cmp(other) == Some(Ordering::Equal)
    }

    pub fn add(self, other: Num) -> R<Num> {
        match (self, other) {
            (I(a), I(b)) => a.checked_add(b).map(I).ok_or_else(|| "integer overflow".to_string()),
            (a, b) => Ok(F(a.f() + b.f())),
        }
    }

    pub fn sub(self, other: Num) -> R<Num> {
        match (self, other) {
            (I(a), I(b)) => a.checked_sub(b).map(I).ok_or_else(|| "integer overflow".to_string()),
            (a, b) => Ok(F(a.f() - b.f())),
        }
    }

    pub fn mul(self, other: Num) -> R<Num> {
        match (self, other) {
            (I(a), I(b)) => a.checked_mul(b).map(I).ok_or_else(|| "integer overflow".to_string()),
            (a, b) => Ok(F(a.f() * b.f())),
        }
    }

    /// True division: always a float, and `ZeroDivisionError` on a zero divisor.
    pub fn div(self, other: Num) -> R<Num> {
        if !other.truthy() {
            return Err("division by zero".to_string());
        }
        Ok(F(self.f() / other.f()))
    }

    pub fn neg(self) -> Num {
        match self {
            I(a) => I(-a),
            F(x) => F(-x),
        }
    }
}

/// `int` against `float`, without rounding the int.
fn int_vs_float(i: i128, f: f64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    const LIMIT: f64 = 1.7014118346046923e38; // 2^127
    if f >= LIMIT {
        return Some(Ordering::Less);
    }
    if f < -LIMIT {
        return Some(Ordering::Greater);
    }
    let whole = f.trunc();
    let frac = f - whole;
    match i.cmp(&(whole as i128)) {
        Ordering::Equal if frac > 0.0 => Some(Ordering::Less),
        Ordering::Equal if frac < 0.0 => Some(Ordering::Greater),
        other => Some(other),
    }
}

/// `min(xs)`: the first element no later element is less than.
pub fn min(xs: &[Num]) -> R<Num> {
    let mut it = xs.iter().copied();
    let mut best = it.next().ok_or("min() of an empty sequence")?;
    for x in it {
        if x.lt(best) {
            best = x;
        }
    }
    Ok(best)
}

/// `max(xs)`: the first element no later element is greater than.
pub fn max(xs: &[Num]) -> R<Num> {
    let mut it = xs.iter().copied();
    let mut best = it.next().ok_or("max() of an empty sequence")?;
    for x in it {
        if x.gt(best) {
            best = x;
        }
    }
    Ok(best)
}

/// `sum(xs)`: ints add exactly while every item is an int; from the first float
/// on, CPython 3.12+ keeps a Neumaier-compensated float sum (ints included).
/// An empty sum is the int 0.
pub fn sum(xs: &[Num]) -> R<Num> {
    let mut i_result: i128 = 0;
    let mut at = 0;
    while let Some(I(v)) = xs.get(at) {
        i_result = i_result.checked_add(*v).ok_or("integer overflow in sum")?;
        at += 1;
    }
    let Some(first) = xs.get(at) else {
        return Ok(I(i_result));
    };
    let mut f = i_result as f64 + first.f();
    let mut c = 0.0f64;
    for item in &xs[at + 1..] {
        let x = item.f();
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    Ok(F(f))
}

/// `statistics.median(xs)`.
pub fn median(xs: &[Num]) -> R<Num> {
    if xs.is_empty() {
        return Err("no median for empty data".to_string());
    }
    let mut data = xs.to_vec();
    data.sort_by(|a, b| a.cmp(*b).unwrap_or(Ordering::Equal));
    let n = data.len();
    if n % 2 == 1 {
        Ok(data[n / 2])
    } else {
        data[n / 2 - 1].add(data[n / 2])?.div(I(2))
    }
}

// ---------------------------------------------------------------------------
// Text formats.
// ---------------------------------------------------------------------------

/// `repr(float)`: the shortest digits that read back to `x`, closest to `x`
/// among those, laid out as Python does (exponent form below 1e-4 and from 1e16).
pub fn repr_f64(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let shortest = format!("{x:e}");
    // Rust's shortest search breaks an exact tie between two equally short
    // candidates upward, Python's to the even digit; the correctly rounded
    // string of the same length is Python's whenever it reads back to `x`.
    let digits = shortest
        .trim_start_matches('-')
        .split('e')
        .next()
        .unwrap_or("")
        .chars()
        .filter(char::is_ascii_digit)
        .count();
    let rounded = format!("{:.*e}", digits.saturating_sub(1), x);
    let sci = if rounded.parse::<f64>().ok() == Some(x) {
        rounded
    } else {
        shortest
    };
    let (neg, sci) = match sci.strip_prefix('-') {
        Some(rest) => (true, rest.to_string()),
        None => (false, sci),
    };
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let decpt = exponent + 1;
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        out.push_str(&"0".repeat((-decpt) as usize));
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(&digits);
        out.push_str(&"0".repeat(decpt as usize - digits.len()));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// `str(n)` / `f"{n}"`.
pub fn str_num(n: Num) -> String {
    match n {
        I(i) => i.to_string(),
        F(x) => repr_f64(x),
    }
}

/// `format(x, ".{prec}f")`.
pub fn fixed(x: f64, prec: usize) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x < 0.0 { "-inf" } else { "inf" }.to_string()
    } else {
        format!("{x:.prec$}")
    }
}

/// `format(x, ".{prec}%")`: the value times 100, fixed, then a percent sign.
pub fn percent(x: f64, prec: usize) -> String {
    format!("{}%", fixed(x * 100.0, prec))
}

/// `format(x, "+.{prec}%")`.
pub fn signed_percent(x: f64, prec: usize) -> String {
    let body = percent(x, prec);
    if body.starts_with('-') {
        body
    } else {
        format!("+{body}")
    }
}

/// Python's `str.isspace` per character.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `str.rstrip()`.
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `str.splitlines()`.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let is_break = matches!(
            c,
            '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}'
        );
        if !is_break {
            continue;
        }
        out.push(&s[start..i]);
        let mut next = i + c.len_utf8();
        if c == '\r' && chars.peek().map(|&(_, n)| n) == Some('\n') {
            chars.next();
            next += 1;
        }
        start = next;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `Path.read_text()`: UTF-8, `\r\n` and `\r` read as `\n`.
pub fn read_text(path: &Path) -> R<String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let text = String::from_utf8(raw).map_err(|e| format!("{} is not UTF-8: {e}", path.display()))?;
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// `Path(p).stem`.
pub fn stem(p: &str) -> String {
    let name = p.rsplit('/').find(|s| !s.is_empty()).unwrap_or("");
    match name.rfind('.') {
        Some(i) if i > 0 => name[..i].to_string(),
        _ => name.to_string(),
    }
}

/// `write_text`.
pub fn write_text(path: &Path, text: &str) -> R<()> {
    std::fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// JSON in and out.
// ---------------------------------------------------------------------------

/// A value as `json.loads` produced it and `json.dumps` writes it.
#[derive(Clone, Debug)]
pub enum Val {
    Null,
    Bool(bool),
    Num(Num),
    Str(String),
    Arr(Vec<Val>),
    Obj(Vec<(String, Val)>),
}

impl Val {
    pub fn obj<const N: usize>(fields: [(&str, Val); N]) -> Val {
        Val::Obj(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn int(n: usize) -> Val {
        Val::Num(I(n as i128))
    }

    pub fn float(x: f64) -> Val {
        Val::Num(F(x))
    }

    pub fn num(n: Num) -> Val {
        Val::Num(n)
    }

    pub fn get(&self, key: &str) -> R<&Val> {
        match self {
            Val::Obj(fields) => fields
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .ok_or_else(|| format!("KeyError: {key:?}")),
            _ => Err(format!("expected an object holding {key:?}")),
        }
    }

    pub fn has(&self, key: &str) -> bool {
        matches!(self, Val::Obj(fields) if fields.iter().any(|(k, _)| k == key))
    }

    pub fn arr(&self) -> R<&[Val]> {
        match self {
            Val::Arr(v) => Ok(v),
            _ => Err("expected an array".to_string()),
        }
    }

    pub fn str(&self) -> R<&str> {
        match self {
            Val::Str(s) => Ok(s),
            _ => Err("expected a string".to_string()),
        }
    }

    pub fn num_val(&self) -> R<Num> {
        match self {
            Val::Num(n) => Ok(*n),
            _ => Err("expected a number".to_string()),
        }
    }

    pub fn nums(&self) -> R<Vec<Num>> {
        self.arr()?.iter().map(Val::num_val).collect()
    }

    /// `str(v)` for a scalar, as an f-string interpolates it.
    pub fn display(&self) -> String {
        match self {
            Val::Null => "None".to_string(),
            Val::Bool(true) => "True".to_string(),
            Val::Bool(false) => "False".to_string(),
            Val::Num(n) => str_num(*n),
            Val::Str(s) => s.clone(),
            other => dumps(other, None),
        }
    }

    /// Python `==` between two loaded values (`1 == 1.0`).
    pub fn same(&self, other: &Val) -> bool {
        match (self, other) {
            (Val::Null, Val::Null) => true,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Num(a), Val::Num(b)) => a.eq(*b),
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Arr(a), Val::Arr(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same(y)),
            (Val::Obj(a), Val::Obj(b)) => {
                a.len() == b.len() && a.iter().all(|(k, v)| b.iter().any(|(k2, v2)| k == k2 && v.same(v2)))
            }
            _ => false,
        }
    }
}

/// `json.loads(text)`.
pub fn parse_json(text: &str, what: &str) -> R<Val> {
    let syntax = crate::json::Syntax {
        what: "JSON",
        max_depth: 64,
        uints_only: false,
        literals: true,
    };
    let parsed = crate::json::parse(text, syntax).map_err(|e| format!("{what}: {e}"))?;
    convert(&parsed).map_err(|e| format!("{what}: {e}"))
}

fn convert(j: &Json) -> R<Val> {
    Ok(match j {
        Json::Null => Val::Null,
        Json::Bool(b) => Val::Bool(*b),
        Json::Str(s) => Val::Str(s.clone()),
        Json::Num {
            form: NumForm::Float,
            value,
            ..
        } => Val::Num(F(*value)),
        Json::Num { .. } => Val::Num(I(j.exact_int().ok_or("an integer too large for this tool")?)),
        Json::Array(items) => Val::Arr(items.iter().map(convert).collect::<R<_>>()?),
        Json::Object(fields) => Val::Obj(
            fields
                .iter()
                .map(|(k, v)| Ok((k.clone(), convert(v)?)))
                .collect::<R<_>>()?,
        ),
    })
}

/// `json.loads(path.read_text())`.
pub fn load_file(path: &Path) -> R<Val> {
    parse_json(&read_text(path)?, &path.display().to_string())
}

fn push_str(out: &mut String, s: &str) {
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
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn push_num(out: &mut String, n: Num) {
    match n {
        I(i) => out.push_str(&i.to_string()),
        F(x) if x.is_nan() => out.push_str("NaN"),
        F(x) if x.is_infinite() => out.push_str(if x < 0.0 { "-Infinity" } else { "Infinity" }),
        F(x) => out.push_str(&repr_f64(x)),
    }
}

fn write_val(out: &mut String, v: &Val, indent: Option<usize>, level: usize) {
    match v {
        Val::Null => out.push_str("null"),
        Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Val::Num(n) => push_num(out, *n),
        Val::Str(s) => push_str(out, s),
        Val::Arr(items) if items.is_empty() => out.push_str("[]"),
        Val::Obj(fields) if fields.is_empty() => out.push_str("{}"),
        Val::Arr(items) => {
            out.push('[');
            for (k, item) in items.iter().enumerate() {
                separator(out, indent, level + 1, k == 0);
                write_val(out, item, indent, level + 1);
            }
            close(out, indent, level, ']');
        }
        Val::Obj(fields) => {
            out.push('{');
            for (k, (key, value)) in fields.iter().enumerate() {
                separator(out, indent, level + 1, k == 0);
                push_str(out, key);
                out.push_str(if indent.is_some() { ": " } else { ":" });
                write_val(out, value, indent, level + 1);
            }
            close(out, indent, level, '}');
        }
    }
}

fn separator(out: &mut String, indent: Option<usize>, level: usize, first: bool) {
    if !first {
        out.push(',');
    }
    if let Some(width) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(width * level));
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize, bracket: char) {
    if let Some(width) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(width * level));
    }
    out.push(bracket);
}

/// `json.dumps(v, indent=N)` when `indent` is given (item separator `,`, key
/// separator `: `), else `json.dumps(v, separators=(",", ":"))`.
pub fn dumps(v: &Val, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_val(&mut out, v, indent, 0);
    out
}

// ---------------------------------------------------------------------------
// Ordered maps: a Python dict keeps first-insertion order and overwrites in place.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct OMap<V> {
    pub items: Vec<(String, V)>,
}

impl<V> Default for OMap<V> {
    fn default() -> Self {
        OMap { items: Vec::new() }
    }
}

impl<V> OMap<V> {
    pub fn get(&self, key: &str) -> Option<&V> {
        self.items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.items.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn set(&mut self, key: &str, value: V) {
        match self.get_mut(key) {
            Some(slot) => *slot = value,
            None => self.items.push((key.to_string(), value)),
        }
    }

    /// `d.setdefault(key, make())`.
    pub fn entry(&mut self, key: &str, make: impl FnOnce() -> V) -> &mut V {
        if self.get(key).is_none() {
            self.items.push((key.to_string(), make()));
        }
        self.get_mut(key).expect("just inserted")
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.items.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.items.iter().map(|(_, v)| v)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// `"x".split(sep, 1)[-1]`: the text after the first `sep`, or all of it.
pub fn after_colon(name: &str) -> &str {
    name.split_once(':').map_or(name, |(_, rest)| rest)
}
