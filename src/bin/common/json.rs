//! JSON adapter for tessl benchmark tools and test manifests,
//! delegating strict parsing to `ojas_io::json`.

#![allow(dead_code)]

use ojas_io::json::{self, JsonNumber, JsonValue};

pub use ojas_io::json::JsonLimits;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    Str(String),
    Num { value: f64, int: u64, form: NumForm },
    Bool(bool),
    Null,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumForm {
    Float,
    Uint,
    NegInt,
    BigInt,
}

impl Json {
    pub fn exact_int(&self) -> Option<i128> {
        match self {
            Json::Num {
                int,
                form: NumForm::Uint,
                ..
            } => Some(i128::from(*int)),
            Json::Num {
                int,
                form: NumForm::NegInt,
                ..
            } => Some(-i128::from(*int)),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Syntax {
    pub what: &'static str,
    pub max_depth: usize,
    pub uints_only: bool,
    pub literals: bool,
}

use std::borrow::Cow;

/// Rewrite `-0` (outside strings) as `0`. Works on bytes and copies every other
/// byte as it was, so UTF-8 text survives unchanged: only ASCII is replaced.
fn normalize_neg_zero(text: &str) -> Cow<'_, str> {
    if !text.contains("-0") {
        return Cow::Borrowed(text);
    }
    let bytes = text.as_bytes();
    let mut in_str = false;
    let mut escape = false;
    let mut i = 0;
    let mut changed = false;
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            out.push(b);
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_str = true;
            out.push(b);
            i += 1;
            continue;
        }
        if b == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'0' {
            let next = bytes.get(i + 2).copied();
            if !matches!(next, Some(b'.' | b'e' | b'E' | b'0'..=b'9')) {
                out.push(b'0');
                i += 2;
                changed = true;
                continue;
            }
        }
        out.push(b);
        i += 1;
    }
    if !changed {
        return Cow::Borrowed(text);
    }
    // Only ASCII bytes were replaced in valid UTF-8, so this holds; were it
    // ever not, the unrewritten text goes to the parser, which refuses it.
    String::from_utf8(out).map_or(Cow::Borrowed(text), Cow::Owned)
}

pub fn parse(text: &str, syntax: Syntax) -> Result<Json, String> {
    let limits = JsonLimits {
        max_depth: syntax.max_depth,
        ..JsonLimits::DEFAULT
    };
    let clean = normalize_neg_zero(text);
    let val = json::parse_json_with(&clean, &limits).map_err(|e| {
        let d = e.detail();
        if d.contains("overflows") {
            format!("{}: an integer too large for this tool: {d}", syntax.what)
        } else {
            format!("{}: {d}", syntax.what)
        }
    })?;
    from_ojas_val(val, syntax)
}

fn from_ojas_val(v: JsonValue, syntax: Syntax) -> Result<Json, String> {
    match v {
        JsonValue::Null => {
            if !syntax.literals {
                return Err(format!("{}: null is not permitted", syntax.what));
            }
            Ok(Json::Null)
        }
        JsonValue::Bool(b) => {
            if !syntax.literals {
                return Err(format!("{}: boolean is not permitted", syntax.what));
            }
            Ok(Json::Bool(b))
        }
        JsonValue::String(s) => Ok(Json::Str(s)),
        JsonValue::Number(num) => match num {
            JsonNumber::U64(n) => Ok(Json::Num {
                value: n as f64,
                int: n,
                form: NumForm::Uint,
            }),
            JsonNumber::I64(n) => {
                if syntax.uints_only {
                    return Err(format!("{}: negative integer not permitted", syntax.what));
                }
                Ok(Json::Num {
                    value: n as f64,
                    int: n.unsigned_abs(),
                    form: NumForm::NegInt,
                })
            }
            JsonNumber::F64(f) => {
                if syntax.uints_only {
                    return Err(format!("{}: floating point number not permitted", syntax.what));
                }
                Ok(Json::Num {
                    value: f,
                    int: 0,
                    form: NumForm::Float,
                })
            }
        },
        JsonValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(from_ojas_val(item, syntax)?);
            }
            Ok(Json::Array(out))
        }
        JsonValue::Object(entries) => {
            let mut out = Vec::with_capacity(entries.len());
            for (k, val) in entries {
                out.push((k, from_ojas_val(val, syntax)?));
            }
            Ok(Json::Object(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTAX: Syntax = Syntax {
        what: "test",
        max_depth: 4,
        uints_only: false,
        literals: true,
    };

    fn string_at(v: &Json, key: &str) -> String {
        match v.get(key) {
            Some(Json::Str(s)) => s.clone(),
            other => panic!("{key}: expected a string, got {other:?}"),
        }
    }

    /// The `-0` rewrite once pushed each byte as a Latin-1 char once it had
    /// triggered, so every non-ASCII string in the document was corrupted.
    #[test]
    fn the_negative_zero_rewrite_keeps_non_ascii_strings_intact() {
        let text = "{\"bos\": \"é日😀\", \"zero\": -0, \"neg\": \"-0\", \"tail\": \"ß\"}";
        let v = parse(text, SYNTAX).unwrap();
        assert_eq!(string_at(&v, "bos"), "é日😀");
        assert_eq!(string_at(&v, "neg"), "-0");
        assert_eq!(string_at(&v, "tail"), "ß");
        assert!(matches!(v.get("zero"), Some(Json::Num { value, .. }) if *value == 0.0));
    }

    #[test]
    fn the_rewrite_changes_only_a_bare_negative_zero() {
        assert_eq!(
            normalize_neg_zero("{\"a\": -0, \"b\": -0.5, \"c\": -01e2}"),
            "{\"a\": 0, \"b\": -0.5, \"c\": -01e2}"
        );
        assert!(matches!(normalize_neg_zero("{\"a\": \"é\"}"), Cow::Borrowed(_)));
    }
}
