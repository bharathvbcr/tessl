//! The one reader of a Hugging Face `config.json` that `bert`, `qwen35_model`
//! and the EmbedGemma2 text, vision and audio configs share: parse to an object, then typed field readers over
//! `ojas-io`'s [`JsonValue`]. Every message names the key and starts `config.json:`.
//!
//! A key that is present with the wrong type is always an error, `null`
//! included: a reader never takes a wrong-typed value for an absent one.

use ojas_io::json::{self, JsonLimits, JsonNumber, JsonValue};

/// Parse `text`; the root must be an object.
pub(crate) fn parse_object(text: &str) -> Result<JsonValue, String> {
    let limits = JsonLimits {
        max_depth: 16,
        ..JsonLimits::DEFAULT
    };
    let root = json::parse_json_with(text, &limits).map_err(|e| format!("config.json: {}", e.detail()))?;
    if !matches!(root, JsonValue::Object(_)) {
        return Err("config.json: the root is not an object".into());
    }
    Ok(root)
}

/// The value of `k`, which must be present.
pub(crate) fn field<'a>(c: &'a JsonValue, k: &str) -> Result<&'a JsonValue, String> {
    c.get(k).ok_or_else(|| format!("config.json: missing {k:?}"))
}

/// A required non-negative integer that fits `u32`.
pub(crate) fn uint(c: &JsonValue, k: &str) -> Result<u32, String> {
    match field(c, k)? {
        JsonValue::Number(JsonNumber::U64(n)) => {
            u32::try_from(*n).map_err(|_| format!("config.json: {k} = {n} exceeds u32"))
        }
        v => Err(format!("config.json: {k} must be a non-negative integer, got {v:?}")),
    }
}

/// A number (already fetched as `v`, which was read from key `k`).
pub(crate) fn float(v: &JsonValue, k: &str) -> Result<f64, String> {
    match v {
        JsonValue::Number(JsonNumber::F64(value)) => Ok(*value),
        JsonValue::Number(JsonNumber::U64(n)) => Ok(*n as f64),
        JsonValue::Number(JsonNumber::I64(n)) => Ok(*n as f64),
        v => Err(format!("config.json: {k} must be a number, got {v:?}")),
    }
}

/// An optional boolean.
pub(crate) fn flag(c: &JsonValue, k: &str) -> Result<Option<bool>, String> {
    match c.get(k) {
        None => Ok(None),
        Some(JsonValue::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(format!("config.json: {k} must be true or false, got {v:?}")),
    }
}

/// An optional string.
pub(crate) fn string<'a>(c: &'a JsonValue, k: &str) -> Result<Option<&'a str>, String> {
    match c.get(k) {
        None => Ok(None),
        Some(JsonValue::String(s)) => Ok(Some(s.as_str())),
        Some(v) => Err(format!("config.json: {k} must be a string, got {v:?}")),
    }
}
