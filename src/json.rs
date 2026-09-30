//! A strict JSON parser for the few JSON files tessl reads: `.safetensors`
//! headers and Hugging Face `config.json`.
//!
//! No dependency, no leniency. Keys are unique, strings are valid UTF-8 with
//! only the RFC 8259 escapes (surrogates paired), control characters are
//! refused, numbers follow the RFC grammar (no leading zeros, `+`, `NaN` or
//! hex), and nothing trails the root value. Each caller also states the
//! [`Syntax`] its format needs, so a `.safetensors` header still refuses a
//! float, a negative number, `true` or `null` anywhere, and nesting deeper
//! than the format uses.

/// A parsed value. Objects keep their keys in file order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    Str(String),
    /// A number, with its exact value when it is a non-negative integer that
    /// fits `u64`.
    Num {
        value: f64,
        uint: Option<u64>,
    },
    Bool(bool),
    Null,
}

impl Json {
    /// The field `key` of an object.
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// What a format allows beyond objects, arrays and strings.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Syntax {
    /// Error-message prefix naming the document, e.g. `"header JSON"`.
    pub what: &'static str,
    /// Containers deeper than this are refused.
    pub max_depth: usize,
    /// Only non-negative integers (no sign, fraction or exponent).
    pub uints_only: bool,
    /// `true`, `false` and `null` are allowed.
    pub literals: bool,
}

/// Parse `text` as one JSON value under `syntax`.
pub(crate) fn parse(text: &str, syntax: Syntax) -> Result<Json, String> {
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
        syntax,
    };
    p.ws();
    let root = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!(
            "{}: trailing bytes after the root value at {}",
            syntax.what, p.i
        ));
    }
    Ok(root)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    syntax: Syntax,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn err<T>(&self, msg: &str) -> Result<T, String> {
        Err(format!("{}: {msg} at byte {}", self.syntax.what, self.i))
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            self.err(&format!("expected {:?}", c as char))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        let unsupported = if self.syntax.uints_only && !self.syntax.literals {
            "unsupported JSON value (only objects, arrays, strings and non-negative integers)"
        } else {
            "unsupported JSON value"
        };
        match self.s.get(self.i) {
            Some(b'{') | Some(b'[') if depth >= self.syntax.max_depth => self.err(&format!(
                "nesting deeper than the format uses ({} levels)",
                self.syntax.max_depth
            )),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'0'..=b'9') => self.number(),
            Some(b'-') if !self.syntax.uints_only => self.number(),
            Some(b't') if self.syntax.literals => self.literal("true", Json::Bool(true)),
            Some(b'f') if self.syntax.literals => self.literal("false", Json::Bool(false)),
            Some(b'n') if self.syntax.literals => self.literal("null", Json::Null),
            Some(_) => self.err(unsupported),
            None => self.err("unexpected end"),
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.s.get(self.i..self.i + word.len()) == Some(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            self.err("unsupported JSON value")
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut out: Vec<(String, Json)> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Object(out));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return self.err("expected a string key");
            }
            let k = self.string()?;
            if !seen.insert(k.clone()) {
                return self.err(&format!("duplicate key {k:?}"));
            }
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value(depth + 1)?;
            out.push((k, v));
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(out));
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Array(out));
        }
        loop {
            self.ws();
            out.push(self.value(depth + 1)?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(out));
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
    }

    /// RFC 8259 `number`: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        let negative = self.s.get(self.i) == Some(&b'-');
        if negative {
            self.i += 1;
        }
        let int_start = self.i;
        let mut uint: Option<u64> = Some(0);
        while let Some(&c @ b'0'..=b'9') = self.s.get(self.i) {
            uint = uint
                .and_then(|v| v.checked_mul(10))
                .and_then(|v| v.checked_add(u64::from(c - b'0')));
            self.i += 1;
        }
        let digits = self.i - int_start;
        if digits == 0 {
            return self.err("a number needs digits");
        }
        if digits > 1 && self.s[int_start] == b'0' {
            return self.err("integer with a leading zero");
        }
        let mut integral = true;
        if self.s.get(self.i) == Some(&b'.') {
            integral = false;
            self.i += 1;
            if !matches!(self.s.get(self.i), Some(b'0'..=b'9')) {
                return self.err("a fraction needs digits");
            }
            while let Some(b'0'..=b'9') = self.s.get(self.i) {
                self.i += 1;
            }
        }
        if let Some(b'e' | b'E') = self.s.get(self.i) {
            integral = false;
            self.i += 1;
            if let Some(b'+' | b'-') = self.s.get(self.i) {
                self.i += 1;
            }
            if !matches!(self.s.get(self.i), Some(b'0'..=b'9')) {
                return self.err("an exponent needs digits");
            }
            while let Some(b'0'..=b'9') = self.s.get(self.i) {
                self.i += 1;
            }
        }
        if self.syntax.uints_only {
            if !integral {
                return self.err("non-integer number");
            }
            if uint.is_none() {
                return Err(format!("{}: integer overflows u64 at byte {start}", self.syntax.what));
            }
        }
        let lexeme = std::str::from_utf8(&self.s[start..self.i]).map_err(|e| e.to_string())?;
        let value: f64 = lexeme
            .parse()
            .map_err(|_| format!("{}: bad number {lexeme:?} at byte {start}", self.syntax.what))?;
        if !value.is_finite() {
            return Err(format!(
                "{}: number {lexeme:?} overflows f64 at byte {start}",
                self.syntax.what
            ));
        }
        let uint = if integral && !negative { uint } else { None };
        Ok(Json::Num { value, uint })
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let Some(&c) = self.s.get(self.i) else {
                return self.err("unterminated string");
            };
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        return self.err("unterminated escape");
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.s.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return self.err("unpaired high surrogate");
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return self.err("invalid low surrogate");
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return self.err("unpaired low surrogate");
                            } else {
                                hi
                            };
                            match char::from_u32(cp) {
                                Some(ch) => out.push(ch),
                                None => return self.err("invalid code point"),
                            }
                        }
                        _ => return self.err("invalid escape"),
                    }
                }
                0x00..=0x1f => return self.err("control character in string"),
                _ => {
                    // Copy the whole UTF-8 sequence. The input is a &str, so a
                    // lead byte is followed by its tail.
                    let start = self.i - 1;
                    let len = match c {
                        0x00..=0x7f => 1,
                        0xc0..=0xdf => 2,
                        0xe0..=0xef => 3,
                        _ => 4,
                    };
                    let end = start + len;
                    let chunk = std::str::from_utf8(&self.s[start..end])
                        .map_err(|_| format!("{}: bad UTF-8 at byte {start}", self.syntax.what))?;
                    out.push_str(chunk);
                    self.i = end;
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let Some(h) = self.s.get(self.i..self.i + 4) else {
            return self.err("short \\u escape");
        };
        // Digits only: `from_str_radix` alone would also take a leading '+'.
        let mut v = 0u32;
        for &c in h {
            let d = (c as char)
                .to_digit(16)
                .ok_or_else(|| format!("{}: bad \\u escape at byte {}", self.syntax.what, self.i))?;
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANY: Syntax = Syntax {
        what: "t",
        max_depth: 8,
        uints_only: false,
        literals: true,
    };

    fn num(text: &str) -> Result<Json, String> {
        parse(text, ANY)
    }

    #[test]
    fn numbers_follow_the_rfc_grammar() {
        for (text, value, uint) in [
            ("0", 0.0, Some(0)),
            ("-0", -0.0, None),
            ("10000000", 1e7, Some(10_000_000)),
            ("1e-06", 1e-6, None),
            ("0.25", 0.25, None),
            ("-1.5E+3", -1500.0, None),
            ("18446744073709551615", 18446744073709551615.0, Some(u64::MAX)),
            ("18446744073709551616", 18446744073709551616.0, None),
        ] {
            assert_eq!(num(text).unwrap(), Json::Num { value, uint }, "{text}");
        }
        for bad in [
            "01", "1.", ".5", "1e", "1e+", "-", "+1", "--1", "1.e3", "0x10", "NaN", "1e400",
        ] {
            assert!(num(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn literals_and_depth_are_per_format() {
        assert_eq!(
            parse("[true,false,null]", ANY).unwrap(),
            Json::Array(vec![Json::Bool(true), Json::Bool(false), Json::Null])
        );
        for bad in ["tru", "nul", "True", "falsey"] {
            assert!(parse(bad, ANY).is_err(), "{bad}");
        }
        let strict = Syntax {
            what: "h",
            max_depth: 2,
            uints_only: true,
            literals: false,
        };
        assert!(parse("true", strict).unwrap_err().contains("unsupported JSON value"));
        assert!(parse("-1", strict).unwrap_err().contains("unsupported JSON value"));
        assert!(parse("1.5", strict).unwrap_err().contains("non-integer"));
        assert!(parse("[[1]]", strict).is_ok());
        assert!(parse("[[[1]]]", strict).unwrap_err().contains("nesting deeper"));
    }

    #[test]
    fn get_finds_object_fields() {
        let v = parse(r#"{"a":{"b":[1]},"c":"d"}"#, ANY).unwrap();
        assert_eq!(v.get("c"), Some(&Json::Str("d".into())));
        assert!(v.get("a").and_then(|a| a.get("b")).is_some());
        assert_eq!(v.get("z"), None);
        assert_eq!(Json::Null.get("a"), None);
    }
}
