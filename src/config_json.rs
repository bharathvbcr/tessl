//! The one reader of a Hugging Face `config.json` that `bert` and
//! `qwen35_model` share: parse to an object, then typed field readers over
//! tessl's own [`Json`]. Every message names the key and starts `config.json:`.
//!
//! A key that is present with the wrong type is always an error, `null`
//! included: a reader never takes a wrong-typed value for an absent one.

use crate::json::{self, Json, NumForm, Syntax};

/// Parse `text`; the root must be an object.
pub(crate) fn parse_object(text: &str) -> Result<Json, String> {
    const SYNTAX: Syntax = Syntax {
        what: "config.json",
        max_depth: 16,
        uints_only: false,
        literals: true,
    };
    let root = json::parse(text, SYNTAX)?;
    if !matches!(root, Json::Object(_)) {
        return Err("config.json: the root is not an object".into());
    }
    Ok(root)
}

/// The value of `k`, which must be present.
pub(crate) fn field<'a>(c: &'a Json, k: &str) -> Result<&'a Json, String> {
    c.get(k).ok_or_else(|| format!("config.json: missing {k:?}"))
}

/// A required non-negative integer that fits `u32`.
pub(crate) fn uint(c: &Json, k: &str) -> Result<u32, String> {
    match field(c, k)? {
        Json::Num {
            form: NumForm::Uint,
            int: n,
            ..
        } => u32::try_from(*n).map_err(|_| format!("config.json: {k} = {n} exceeds u32")),
        v => Err(format!("config.json: {k} must be a non-negative integer, got {v:?}")),
    }
}

/// A number (already fetched as `v`, which was read from key `k`).
pub(crate) fn float(v: &Json, k: &str) -> Result<f64, String> {
    match v {
        Json::Num { value, .. } => Ok(*value),
        v => Err(format!("config.json: {k} must be a number, got {v:?}")),
    }
}

/// An optional boolean.
pub(crate) fn flag(c: &Json, k: &str) -> Result<Option<bool>, String> {
    match c.get(k) {
        None => Ok(None),
        Some(Json::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(format!("config.json: {k} must be true or false, got {v:?}")),
    }
}

/// An optional string.
pub(crate) fn string<'a>(c: &'a Json, k: &str) -> Result<Option<&'a str>, String> {
    match c.get(k) {
        None => Ok(None),
        Some(Json::Str(s)) => Ok(Some(s.as_str())),
        Some(v) => Err(format!("config.json: {k} must be a string, got {v:?}")),
    }
}
