//! Reading the result JSON files. Every accessor carries the path it came from,
//! so a missing key, a wrong type or an empty list stops the figure with a
//! message such as `summary[2].control.ms_min_all: missing key` instead of a
//! panic or a quietly empty plot.

use crate::json::{Json, Syntax};
use std::path::Path;

pub type R<T> = Result<T, String>;

const SYNTAX: Syntax = Syntax {
    what: "result JSON",
    max_depth: 64,
    uints_only: false,
    literals: true,
};

/// Read and parse one result file.
pub fn parse_file(path: &Path) -> R<Json> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    crate::json::parse(&text, SYNTAX).map_err(|e| format!("{}: {e}", path.display()))
}

/// A value of a parsed document together with where it sits in it.
#[derive(Clone)]
pub struct N<'a> {
    v: &'a Json,
    path: String,
}

impl<'a> N<'a> {
    pub fn root(v: &'a Json, name: &str) -> N<'a> {
        N {
            v,
            path: name.to_string(),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Field `key` of an object; an error naming the path when it is absent.
    pub fn get(&self, key: &str) -> R<N<'a>> {
        match self.v {
            Json::Object(_) => self
                .v
                .get(key)
                .map(|v| N {
                    v,
                    path: format!("{}.{key}", self.path),
                })
                .ok_or_else(|| format!("{}.{key}: missing key", self.path)),
            _ => Err(format!("{}: expected an object with key {key:?}", self.path)),
        }
    }

    /// Field `key` when present and not `null`.
    pub fn opt(&self, key: &str) -> Option<N<'a>> {
        match self.v.get(key) {
            None | Some(Json::Null) => None,
            Some(v) => Some(N {
                v,
                path: format!("{}.{key}", self.path),
            }),
        }
    }

    /// The elements of an array.
    pub fn items(&self) -> R<Vec<N<'a>>> {
        match self.v {
            Json::Array(a) => Ok(a
                .iter()
                .enumerate()
                .map(|(i, v)| N {
                    v,
                    path: format!("{}[{i}]", self.path),
                })
                .collect()),
            _ => Err(format!("{}: expected an array", self.path)),
        }
    }

    /// The elements of an array that must not be empty.
    pub fn nonempty(&self) -> R<Vec<N<'a>>> {
        let items = self.items()?;
        if items.is_empty() {
            return Err(format!("{}: empty list, nothing to plot", self.path));
        }
        Ok(items)
    }

    /// The members of an object, in file order.
    pub fn entries(&self) -> R<Vec<(String, N<'a>)>> {
        match self.v {
            Json::Object(fields) => Ok(fields
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        N {
                            v,
                            path: format!("{}.{k}", self.path),
                        },
                    )
                })
                .collect()),
            _ => Err(format!("{}: expected an object", self.path)),
        }
    }

    /// A finite number.
    pub fn f(&self) -> R<f64> {
        match self.v {
            Json::Num { value, .. } if value.is_finite() => Ok(*value),
            Json::Num { .. } => Err(format!("{}: number is not finite", self.path)),
            _ => Err(format!("{}: expected a number", self.path)),
        }
    }

    /// A finite number above zero (a time or a rate that goes on a log axis).
    pub fn pos(&self) -> R<f64> {
        let v = self.f()?;
        if v > 0.0 {
            Ok(v)
        } else {
            Err(format!("{}: {v} is not above zero", self.path))
        }
    }

    pub fn s(&self) -> R<&'a str> {
        match self.v {
            Json::Str(s) => Ok(s),
            _ => Err(format!("{}: expected a string", self.path)),
        }
    }

    /// A non-empty array of finite numbers.
    pub fn floats(&self) -> R<Vec<f64>> {
        self.nonempty()?.iter().map(|n| n.f()).collect()
    }

    /// A non-empty array of numbers above zero.
    pub fn positives(&self) -> R<Vec<f64>> {
        self.nonempty()?.iter().map(|n| n.pos()).collect()
    }

    /// The element of an array whose field `key` is the string `want`.
    pub fn find(&self, key: &str, want: &str) -> R<N<'a>> {
        for item in self.items()? {
            if item.get(key)?.s()? == want {
                return Ok(item);
            }
        }
        Err(format!("{}: no element with {key} = {want:?}", self.path))
    }
}

/// The smallest of a non-empty list of finite numbers.
pub fn min_of(vals: &[f64]) -> R<f64> {
    vals.iter()
        .copied()
        .reduce(f64::min)
        .ok_or_else(|| "minimum of an empty list".to_string())
}

/// The largest of a non-empty list of finite numbers.
pub fn max_of(vals: &[f64]) -> R<f64> {
    vals.iter()
        .copied()
        .reduce(f64::max)
        .ok_or_else(|| "maximum of an empty list".to_string())
}
