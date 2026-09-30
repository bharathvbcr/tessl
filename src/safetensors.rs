//! Minimal, strict reader for the `.safetensors` checkpoint format.
//!
//! This exists so a Qwen3.5 checkpoint can be loaded straight from the
//! Hugging Face cache, without a second multi-gigabyte copy in another format
//! and without a new dependency. A `.safetensors` file is an 8-byte
//! little-endian header length `n`, then `n` bytes of UTF-8 JSON, then the data
//! buffer. The JSON maps each tensor name to
//! `{"dtype": .., "shape": [..], "data_offsets": [begin, end]}`, offsets being
//! relative to the start of the data buffer; an optional `"__metadata__"` entry
//! maps strings to strings.
//!
//! Everything outside that shape is an error rather than a best-effort parse:
//! a loader that silently read the wrong bytes would produce a model that runs
//! and answers wrongly. In particular this rejects
//!
//! * a header length that is zero, above [`MAX_HEADER_BYTES`], or past the end
//!   of the file;
//! * any JSON beyond objects, arrays, strings and non-negative integers (no
//!   floats, `true`/`false`/`null`, or nesting deeper than the format uses),
//!   duplicate keys, unknown fields, and trailing bytes other than spaces;
//! * a shape whose element count, or byte size, overflows;
//! * `data_offsets` that are reversed, out of the buffer, disagree with
//!   `shape` x dtype size, overlap another tensor, or leave a hole: like the
//!   reference implementation, the tensors must tile the data buffer exactly.
//!
//! Tensor data is read on demand with positioned reads, so opening a file costs
//! only its header.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Largest header accepted. The format's reference implementation caps it at
/// 100 MB; a 2B checkpoint's header is about 90 KB.
pub const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Element types the format defines. Only [`Dtype::F32`], [`Dtype::F16`] and
/// [`Dtype::BF16`] can be read back as numbers here; the rest are parsed so a
/// file holding them still opens and their byte ranges are still checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "BOOL" => Dtype::Bool,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "I16" => Dtype::I16,
            "U16" => Dtype::U16,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I32" => Dtype::I32,
            "U32" => Dtype::U32,
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "I64" => Dtype::I64,
            "U64" => Dtype::U64,
            other => return Err(format!("unsupported dtype {other:?}")),
        })
    }

    /// Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        }
    }
}

/// One tensor's entry in the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range within the data buffer.
    pub begin: u64,
    pub end: u64,
}

impl TensorInfo {
    /// Product of `shape` (1 for a scalar).
    pub fn numel(&self) -> usize {
        // `open` checked this product fits.
        self.shape.iter().product()
    }
}

/// An open `.safetensors` file: its parsed header and where its data starts.
#[derive(Debug)]
pub struct SafeTensors {
    path: PathBuf,
    /// The handle whose header was validated; every read goes through it.
    file: File,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, String>,
}

impl SafeTensors {
    /// Open and validate `path`'s header. No tensor data is read.
    pub fn open(path: &Path) -> Result<Self, String> {
        let what = path.display().to_string();
        let mut f = File::open(path).map_err(|e| format!("{what}: {e}"))?;
        let file_len = f
            .metadata()
            .map_err(|e| format!("{what}: {e}"))?
            .len();
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)
            .map_err(|e| format!("{what}: header length: {e}"))?;
        let n = u64::from_le_bytes(len8);
        if n == 0 || n > MAX_HEADER_BYTES {
            return Err(format!(
                "{what}: header length {n} outside 1..={MAX_HEADER_BYTES}"
            ));
        }
        if n > file_len - 8 {
            return Err(format!(
                "{what}: header length {n} runs past the end of a {file_len}-byte file"
            ));
        }
        let mut header = vec![0u8; n as usize];
        f.read_exact(&mut header)
            .map_err(|e| format!("{what}: header: {e}"))?;
        let data_start = 8 + n;
        let (tensors, metadata) = parse_header(&header, file_len - data_start)
            .map_err(|e| format!("{what}: {e}"))?;
        Ok(Self {
            path: path.to_path_buf(),
            file: f,
            data_start,
            tensors,
            metadata,
        })
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The `__metadata__` entries, if any.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("{}: no tensor {name:?}", self.path.display()))
    }

    fn read_bytes(&self, name: &str, info: &TensorInfo) -> Result<Vec<u8>, String> {
        let len = usize::try_from(info.end - info.begin)
            .map_err(|_| format!("{name}: tensor too large for this host"))?;
        let mut bytes = vec![0u8; len];
        // From the handle `open` validated, never the path again: a file
        // replaced at that path since would be read with this header's
        // offsets. A truncation of this same file fails here, not as short data.
        self.file
            .read_exact_at(&mut bytes, self.data_start + info.begin)
            .map_err(|e| format!("{name}: data: {e}"))?;
        Ok(bytes)
    }

    /// A bf16 tensor's raw bit patterns and shape. Any other dtype is an error:
    /// narrowing is the caller's decision, not the loader's.
    pub fn read_bf16_bits(&self, name: &str) -> Result<(Vec<usize>, Vec<u16>), String> {
        let info = self.info(name)?;
        if info.dtype != Dtype::BF16 {
            return Err(format!("{name}: expected BF16, found {:?}", info.dtype));
        }
        let bytes = self.read_bytes(name, info)?;
        let bits = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok((info.shape.clone(), bits))
    }

    /// A floating tensor widened exactly to f32 (F32, BF16 or F16; every value
    /// of the narrower types is representable in f32).
    pub fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let info = self.info(name)?;
        let bytes = self.read_bytes(name, info)?;
        let data = match info.dtype {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|c| crate::tensor::bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| crate::tensor::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => return Err(format!("{name}: cannot read {other:?} as f32")),
        };
        Ok((info.shape.clone(), data))
    }
}

type Header = (BTreeMap<String, TensorInfo>, BTreeMap<String, String>);

/// Parse and validate the header against a data buffer of `data_len` bytes.
fn parse_header(bytes: &[u8], data_len: u64) -> Result<Header, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("header is not UTF-8: {e}"))?;
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let root = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing bytes after the header object at {}", p.i));
    }
    let Json::Object(entries) = root else {
        return Err("header is not a JSON object".into());
    };
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (name, v) in entries {
        if name == "__metadata__" {
            let Json::Object(m) = v else {
                return Err("__metadata__ is not an object".into());
            };
            for (k, v) in m {
                let Json::Str(s) = v else {
                    return Err(format!("__metadata__ value for {k:?} is not a string"));
                };
                metadata.insert(k, s);
            }
            continue;
        }
        let info = tensor_info(&name, v)?;
        tensors.insert(name, info);
    }

    // Each tensor's bytes are exactly its shape times its dtype size, and the
    // tensors tile [0, data_len) with no gap and no overlap.
    let mut ranges: Vec<(u64, u64, &str)> = Vec::with_capacity(tensors.len());
    for (name, t) in &tensors {
        if t.begin > t.end || t.end > data_len {
            return Err(format!(
                "{name}: data_offsets [{}, {}] outside the {data_len}-byte data buffer",
                t.begin, t.end
            ));
        }
        let numel = t
            .shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
            .ok_or_else(|| format!("{name}: element count overflows"))?;
        usize::try_from(numel).map_err(|_| format!("{name}: element count overflows usize"))?;
        let want = numel
            .checked_mul(t.dtype.size())
            .ok_or_else(|| format!("{name}: byte size overflows"))?;
        if t.end - t.begin != want {
            return Err(format!(
                "{name}: {} bytes in data_offsets, but shape {:?} x {:?} is {want}",
                t.end - t.begin,
                t.shape,
                t.dtype
            ));
        }
        ranges.push((t.begin, t.end, name));
    }
    ranges.sort_unstable();
    let mut at = 0u64;
    for (b, e, name) in ranges {
        if b != at {
            return Err(format!(
                "{name}: starts at byte {b}, expected {at} (tensors must tile the data buffer \
                 with no gap or overlap)"
            ));
        }
        at = e;
    }
    if at != data_len {
        return Err(format!(
            "the tensors cover {at} of {data_len} data bytes; the rest is unindexed"
        ));
    }
    Ok((tensors, metadata))
}

fn tensor_info(name: &str, v: Json) -> Result<TensorInfo, String> {
    let Json::Object(fields) = v else {
        return Err(format!("{name}: entry is not an object"));
    };
    let (mut dtype, mut shape, mut offsets) = (None, None, None);
    for (k, v) in fields {
        match (k.as_str(), v) {
            ("dtype", Json::Str(s)) => dtype = Some(Dtype::parse(&s).map_err(|e| format!("{name}: {e}"))?),
            ("shape", Json::Array(a)) => {
                let dims = a
                    .into_iter()
                    .map(|d| match d {
                        Json::UInt(n) => usize::try_from(n)
                            .map_err(|_| format!("{name}: dimension {n} overflows usize")),
                        _ => Err(format!("{name}: shape holds a non-integer")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shape = Some(dims);
            }
            ("data_offsets", Json::Array(a)) => match a.as_slice() {
                [Json::UInt(b), Json::UInt(e)] => offsets = Some((*b, *e)),
                _ => return Err(format!("{name}: data_offsets is not [begin, end]")),
            },
            (k, _) => return Err(format!("{name}: unexpected or mistyped field {k:?}")),
        }
    }
    let (Some(dtype), Some(shape), Some((begin, end))) = (dtype, shape, offsets) else {
        return Err(format!("{name}: needs dtype, shape and data_offsets"));
    };
    Ok(TensorInfo {
        dtype,
        shape,
        begin,
        end,
    })
}

/// The JSON subset the format uses.
enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    Str(String),
    UInt(u64),
}

/// Header → tensor entry → `shape` / `data_offsets` array: three levels.
const MAX_DEPTH: usize = 3;

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn err<T>(&self, msg: &str) -> Result<T, String> {
        Err(format!("header JSON: {msg} at byte {}", self.i))
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
        match self.s.get(self.i) {
            Some(b'{') | Some(b'[') if depth >= MAX_DEPTH => self.err("nesting deeper than the format uses"),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'0'..=b'9') => Ok(Json::UInt(self.uint()?)),
            Some(_) => self.err("unsupported JSON value (only objects, arrays, strings and non-negative integers)"),
            None => self.err("unexpected end"),
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

    /// A non-negative integer: no sign, fraction, exponent or leading zero.
    fn uint(&mut self) -> Result<u64, String> {
        let start = self.i;
        let mut v: u64 = 0;
        while let Some(&c @ b'0'..=b'9') = self.s.get(self.i) {
            v = v
                .checked_mul(10)
                .and_then(|v| v.checked_add(u64::from(c - b'0')))
                .ok_or_else(|| format!("header JSON: integer overflows u64 at byte {start}"))?;
            self.i += 1;
        }
        if self.i - start > 1 && self.s[start] == b'0' {
            return self.err("integer with a leading zero");
        }
        if let Some(b'.' | b'e' | b'E') = self.s.get(self.i) {
            return self.err("non-integer number");
        }
        Ok(v)
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
                    // Copy the whole UTF-8 sequence; the header was checked to
                    // be valid UTF-8, so a lead byte is followed by its tail.
                    let start = self.i - 1;
                    let len = match c {
                        0x00..=0x7f => 1,
                        0xc0..=0xdf => 2,
                        0xe0..=0xef => 3,
                        _ => 4,
                    };
                    let end = start + len;
                    let chunk = std::str::from_utf8(&self.s[start..end])
                        .map_err(|_| format!("header JSON: bad UTF-8 at byte {start}"))?;
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
                .ok_or_else(|| format!("header JSON: bad \\u escape at byte {}", self.i))?;
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }
}
