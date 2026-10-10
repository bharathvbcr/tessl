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
use std::path::Path;

use crate::json::{self, Json, Syntax};
use crate::plain::{self, PlainScalar};

/// Largest header accepted. The format's reference implementation caps it at
/// 100 MB; a 2B checkpoint's header is about 90 KB.
pub const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Element types the format defines in whole bytes. [`Dtype::F32`],
/// [`Dtype::F16`] and [`Dtype::BF16`] read back as f32
/// ([`SafeTensors::read_f32`]); [`Dtype::U8`], [`Dtype::I8`] and
/// [`Dtype::U32`] (int8 and MLX-packed Q4 weights) as integers; every dtype,
/// fp8 included, as its raw bytes ([`SafeTensors::read_raw`]). The format's
/// sub-byte types (`F4`, `F6_E2M3`, `F6_E3M2`) are refused: their sizes are
/// not whole bytes, so the byte-range check below cannot be stated for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    /// `F8_E4M3`: fp8, 4 exponent bits, 3 mantissa bits (`float8_e4m3fn`).
    F8E4M3,
    /// `F8_E4M3FNUZ`: as [`Dtype::F8E4M3`] with no negative zero.
    F8E4M3Fnuz,
    /// `F8_E5M2`: fp8, 5 exponent bits, 2 mantissa bits.
    F8E5M2,
    /// `F8_E5M2FNUZ`: as [`Dtype::F8E5M2`] with no negative zero.
    F8E5M2Fnuz,
    /// `F8_E8M0`: an 8-bit power-of-two scale (the MX formats' block scale).
    F8E8M0,
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
    /// Complex64: two f32s.
    C64,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "BOOL" => Dtype::Bool,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "F8_E4M3" => Dtype::F8E4M3,
            "F8_E4M3FNUZ" => Dtype::F8E4M3Fnuz,
            "F8_E5M2" => Dtype::F8E5M2,
            "F8_E5M2FNUZ" => Dtype::F8E5M2Fnuz,
            "F8_E8M0" => Dtype::F8E8M0,
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
            "C64" => Dtype::C64,
            other => return Err(format!("unsupported dtype {other:?}")),
        })
    }

    /// Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            Dtype::Bool
            | Dtype::U8
            | Dtype::I8
            | Dtype::F8E4M3
            | Dtype::F8E4M3Fnuz
            | Dtype::F8E5M2
            | Dtype::F8E5M2Fnuz
            | Dtype::F8E8M0 => 1,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::F64 | Dtype::I64 | Dtype::U64 | Dtype::C64 => 8,
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

/// Where a [`SafeTensors`]' bytes live.
#[derive(Debug)]
enum Source {
    /// The handle whose header was validated; every read goes through it.
    File(File),
    /// The whole serialized form, header included.
    Bytes(Vec<u8>),
}

/// An open `.safetensors` file (or its bytes): its parsed header and where its
/// data starts.
#[derive(Debug)]
pub struct SafeTensors {
    /// The path, or the label [`SafeTensors::from_bytes`] was given: what
    /// errors name.
    what: String,
    source: Source,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, String>,
}

/// The header length an 8-byte prefix declares, checked against the total
/// length of a file or buffer.
fn header_len(what: &str, len8: [u8; 8], total: u64) -> Result<u64, String> {
    let n = u64::from_le_bytes(len8);
    if n == 0 || n > MAX_HEADER_BYTES {
        return Err(format!("{what}: header length {n} outside 1..={MAX_HEADER_BYTES}"));
    }
    if n > total - 8 {
        return Err(format!(
            "{what}: header length {n} runs past the end of a {total}-byte file"
        ));
    }
    Ok(n)
}

impl SafeTensors {
    /// Open and validate `path`'s header. No tensor data is read.
    pub fn open(path: &Path) -> Result<Self, String> {
        let what = path.display().to_string();
        let mut f = File::open(path).map_err(|e| format!("{what}: {e}"))?;
        let file_len = f.metadata().map_err(|e| format!("{what}: {e}"))?.len();
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)
            .map_err(|e| format!("{what}: header length: {e}"))?;
        let n = header_len(&what, len8, file_len)?;
        let mut header = vec![0u8; n as usize];
        f.read_exact(&mut header).map_err(|e| format!("{what}: header: {e}"))?;
        let data_start = 8 + n;
        let (tensors, metadata) = parse_header(&header, file_len - data_start).map_err(|e| format!("{what}: {e}"))?;
        Ok(Self {
            what,
            source: Source::File(f),
            data_start,
            tensors,
            metadata,
        })
    }

    /// Validate `bytes`, a whole `.safetensors` serialization held in memory,
    /// exactly as [`Self::open`] validates a file; errors name `what`. For
    /// checkpoints built on the host (tests, conversions) that never need to
    /// touch the file system.
    pub fn from_bytes(what: &str, bytes: Vec<u8>) -> Result<Self, String> {
        let mut len8 = [0u8; 8];
        len8.copy_from_slice(
            bytes
                .get(..8)
                .ok_or_else(|| format!("{what}: header length: {} bytes, need at least 8", bytes.len()))?,
        );
        let total = bytes.len() as u64;
        let n = header_len(what, len8, total)?;
        let data_start = 8 + n;
        let (tensors, metadata) =
            parse_header(&bytes[8..data_start as usize], total - data_start).map_err(|e| format!("{what}: {e}"))?;
        Ok(Self {
            what: what.to_string(),
            source: Source::Bytes(bytes),
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
            .ok_or_else(|| format!("{}: no tensor {name:?}", self.what))
    }

    /// Fill `dst` with `info`'s bytes from one positioned read into `dst`'s
    /// own storage: no staging copy, whatever the tensor's size.
    fn read_into<T: PlainScalar>(&self, name: &str, info: &TensorInfo, dst: &mut [T]) -> Result<(), String> {
        let bytes = plain::bytes_mut(dst);
        if bytes.len() as u64 != info.end - info.begin {
            return Err(format!(
                "{name}: {} destination bytes for a {}-byte tensor",
                bytes.len(),
                info.end - info.begin
            ));
        }
        let start = self.data_start + info.begin;
        match &self.source {
            // From the handle `open` validated, never the path again: a file
            // replaced at that path since would be read with this header's
            // offsets. A truncation of this same file fails here, not as
            // short data.
            Source::File(f) => f
                .read_exact_at(bytes, start)
                .map_err(|e| format!("{name}: data: {e}"))?,
            // `from_bytes` checked that every tensor lies inside the buffer.
            Source::Bytes(b) => {
                let src = usize::try_from(start)
                    .ok()
                    .and_then(|s| b.get(s..s.checked_add(bytes.len())?))
                    .ok_or_else(|| format!("{name}: data: range past the {}-byte buffer", b.len()))?;
                bytes.copy_from_slice(src);
            }
        }
        plain::le_to_native(dst);
        Ok(())
    }

    /// `name`'s elements as `T`, refusing any dtype but `want`.
    fn read_typed<T: PlainScalar + Default>(&self, name: &str, want: Dtype) -> Result<(Vec<usize>, Vec<T>), String> {
        let info = self.info(name)?;
        if info.dtype != want {
            return Err(format!("{name}: expected {want:?}, found {:?}", info.dtype));
        }
        let mut data = vec![T::default(); info.numel()];
        self.read_into(name, info, &mut data)?;
        Ok((info.shape.clone(), data))
    }

    /// A tensor's header entry and its bytes exactly as stored (little-endian
    /// elements), whatever its dtype: the way to read fp8, or any type this
    /// module has no typed reader for.
    pub fn read_raw(&self, name: &str) -> Result<(TensorInfo, Vec<u8>), String> {
        let info = self.info(name)?;
        let len =
            usize::try_from(info.end - info.begin).map_err(|_| format!("{name}: tensor too large for this host"))?;
        let mut bytes = vec![0u8; len];
        self.read_into(name, info, &mut bytes)?;
        Ok((info.clone(), bytes))
    }

    /// A U8 tensor and its shape. Any other dtype is an error.
    pub fn read_u8(&self, name: &str) -> Result<(Vec<usize>, Vec<u8>), String> {
        self.read_typed(name, Dtype::U8)
    }

    /// An I8 tensor (int8 weights) and its shape. Any other dtype is an error.
    pub fn read_i8(&self, name: &str) -> Result<(Vec<usize>, Vec<i8>), String> {
        self.read_typed(name, Dtype::I8)
    }

    /// A U32 tensor (MLX packs eight 4-bit weights per element) and its shape.
    /// Any other dtype is an error.
    pub fn read_u32(&self, name: &str) -> Result<(Vec<usize>, Vec<u32>), String> {
        self.read_typed(name, Dtype::U32)
    }

    /// A bf16 tensor's raw bit patterns and shape. Any other dtype is an error:
    /// narrowing is the caller's decision, not the loader's.
    pub fn read_bf16_bits(&self, name: &str) -> Result<(Vec<usize>, Vec<u16>), String> {
        self.read_typed(name, Dtype::BF16)
    }

    /// [`Self::read_bf16_bits`] into `dst` (a device tensor's shared storage,
    /// say) instead of a new `Vec`, after checking the dtype and `shape`.
    pub(crate) fn read_bf16_bits_into(&self, name: &str, shape: &[usize], dst: &mut [u16]) -> Result<(), String> {
        let info = self.info(name)?;
        if info.dtype != Dtype::BF16 {
            return Err(format!("{name}: expected BF16, found {:?}", info.dtype));
        }
        if info.shape != shape {
            return Err(format!("{name}: shape {:?}, expected {shape:?}", info.shape));
        }
        self.read_into(name, info, dst)
    }

    /// A floating tensor widened exactly to f32 (F32, BF16 or F16; every value
    /// of the narrower types is representable in f32).
    ///
    /// The result is the only allocation: a 16-bit tensor is read into the
    /// upper half of its own f32 storage and widened in place, front to back.
    /// Element `i` widens into bytes `[4i, 4i + 4)` while the unread elements
    /// `j > i` sit at `[2n + 2j, ..)`, at or past `2n + 2i + 2 >= 4i + 4` for
    /// every `i < n`, so no write lands on a value not yet read.
    pub fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let info = self.info(name)?;
        let mut data = vec![0f32; info.numel()];
        self.read_f32_into(name, &info.shape, &mut data)?;
        Ok((info.shape.clone(), data))
    }

    /// [`Self::read_f32`] into `dst` (a device tensor's shared storage, say)
    /// instead of a new `Vec`, after checking the dtype and `shape`; a 16-bit
    /// tensor is widened in place inside `dst` as described there.
    pub(crate) fn read_f32_into(&self, name: &str, shape: &[usize], dst: &mut [f32]) -> Result<(), String> {
        let info = self.info(name)?;
        let narrow = match info.dtype {
            Dtype::F32 => false,
            Dtype::BF16 | Dtype::F16 => true,
            other => return Err(format!("{name}: cannot read {other:?} as f32")),
        };
        if info.shape != shape {
            return Err(format!("{name}: shape {:?}, expected {shape:?}", info.shape));
        }
        if !narrow {
            return self.read_into(name, info, dst);
        }
        let n = dst.len();
        let bytes = plain::bytes_mut(dst);
        let (_, upper) = bytes.split_at_mut(2 * n);
        self.read_into(name, info, upper)?;
        // Monomorphized per dtype, so the widen inlines into the loop instead
        // of being an indirect call per element.
        if info.dtype == Dtype::BF16 {
            widen_in_place(bytes, n, crate::tensor::bf16_bits_to_f32);
        } else {
            widen_in_place(bytes, n, crate::tensor::f16_bits_to_f32);
        }
        Ok(())
    }
}

/// Widen the `n` little-endian 16-bit values at `bytes[2n..4n]` into the `n`
/// native-endian f32s at `bytes[..4n]`, front to back (see
/// [`SafeTensors::read_f32`] for why that order never overwrites an unread
/// value).
///
/// Whole blocks of `B` elements go first, over slices the borrow checker can
/// see are disjoint, which lets the loop run without per-element bounds
/// checks: block `i..i + B` writes `[4i, 4i + 4B)` and reads from `2n + 2i`
/// on, and `4i + 4B <= 2n + 2i` exactly when `i + 2B <= n`. The remaining
/// fewer than `2B` elements take the element-at-a-time loop.
fn widen_in_place(bytes: &mut [u8], n: usize, widen: impl Fn(u16) -> f32) {
    const B: usize = 1024;
    let mut i = 0;
    while i + 2 * B <= n {
        let (lo, hi) = bytes.split_at_mut(2 * n + 2 * i);
        let dst = &mut lo[4 * i..4 * (i + B)];
        let src = &hi[..2 * B];
        for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(2)) {
            d.copy_from_slice(&widen(u16::from_le_bytes([s[0], s[1]])).to_ne_bytes());
        }
        i += B;
    }
    for i in i..n {
        let bits = u16::from_le_bytes([bytes[2 * n + 2 * i], bytes[2 * n + 2 * i + 1]]);
        bytes[4 * i..4 * i + 4].copy_from_slice(&widen(bits).to_ne_bytes());
    }
}

/// The JSON subset the format uses: non-negative integers, no literals, and
/// header -> tensor entry -> `shape` / `data_offsets` array, three levels.
const HEADER_SYNTAX: Syntax = Syntax {
    what: "header JSON",
    max_depth: 3,
    uints_only: true,
    literals: false,
};

type Header = (BTreeMap<String, TensorInfo>, BTreeMap<String, String>);

/// Parse and validate the header against a data buffer of `data_len` bytes.
fn parse_header(bytes: &[u8], data_len: u64) -> Result<Header, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("header is not UTF-8: {e}"))?;
    let root = json::parse(text, HEADER_SYNTAX)?;
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
                        Json::Num { uint: Some(n), .. } => {
                            usize::try_from(n).map_err(|_| format!("{name}: dimension {n} overflows usize"))
                        }
                        _ => Err(format!("{name}: shape holds a non-integer")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                shape = Some(dims);
            }
            ("data_offsets", Json::Array(a)) => match a.as_slice() {
                [Json::Num { uint: Some(b), .. }, Json::Num { uint: Some(e), .. }] => offsets = Some((*b, *e)),
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
