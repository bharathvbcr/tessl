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
        plain::checked_numel(&self.shape).expect("open checked the element count fits")
    }
}

/// What identifies a file's contents at one moment: size, modification time
/// (to the nanosecond), inode and device.
///
/// Decision (card ft-51395822): `open` records it and every read re-checks it
/// on the same handle (`fstat`, so no path swap matters), and a mismatch is a
/// "changed since it was opened" error. A file rewritten in place after `open`
/// would otherwise be read with the old header's offsets and yield plausible
/// wrong weights. Limits and consequences, by design:
///
/// * a rename-replace at the path is not detected: the handle still names the
///   old inode, so reads keep serving that file's own consistent bytes;
/// * a metadata-only change (`touch`) changes the mtime and so refuses the next
///   read: it fails closed rather than guessing the contents are unchanged;
/// * a rewrite that restores the mtime (`utimes`) is not detected;
/// * the stamp is checked both before and after each read, which narrows a
///   rewrite racing a read to one entirely inside one nanosecond tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    mtime: (i64, i64),
    ino: u64,
    dev: u64,
}

impl FileStamp {
    fn of(f: &File) -> std::io::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let md = f.metadata()?;
        Ok(Self {
            len: md.len(),
            mtime: (md.mtime(), md.mtime_nsec()),
            ino: md.ino(),
            dev: md.dev(),
        })
    }

    /// Err if `f` no longer matches `self`.
    fn recheck(&self, f: &File, what: &str) -> Result<(), String> {
        let now = Self::of(f).map_err(|e| format!("{what}: {e}"))?;
        if now != *self {
            return Err(format!(
                "{what}: file changed since it was opened (was {self:?}, now {now:?}); refusing to read it with the old header's offsets"
            ));
        }
        Ok(())
    }
}

/// Where a [`SafeTensors`]' bytes live.
#[derive(Debug)]
enum Source {
    /// The handle whose header was validated, and its identity at `open`;
    /// every read goes through the handle after re-checking the identity.
    File(File, FileStamp),
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
    /// Bytes after the header: exactly the tensors' total, as they tile it.
    data_len: u64,
    /// The caller's payload limit (`None`: physical memory), applied again to
    /// each read's own allocation.
    limit: Option<u64>,
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
    ///
    /// The payload budget defaults to this machine's physical memory
    /// (`sysconf(_SC_PHYS_PAGES) * _SC_PAGESIZE`): a file declaring more
    /// tensor bytes is refused at open, before any payload allocation. To use
    /// another budget, call [`Self::open_with_limit`] with `Some(bytes)`.
    pub fn open(path: &Path) -> Result<Self, String> {
        Self::open_with_limit(path, None)
    }

    /// [`Self::open`], refusing before any payload allocation a file whose
    /// declared tensor bytes exceed `limit`, or physical memory when `None`.
    pub fn open_with_limit(path: &Path, limit: Option<u64>) -> Result<Self, String> {
        let what = path.display().to_string();
        // `open_regular_file` checks the handle is a regular file; its length
        // is not used. The stamp's single fstat is the one source of both the
        // length the header is validated against and the identity every read
        // rechecks, so a rewrite between two fstats cannot leave them
        // disagreeing.
        let (mut f, _) = plain::open_regular_file(path)?;
        let stamp = FileStamp::of(&f).map_err(|e| format!("{what}: {e}"))?;
        let file_len = stamp.len;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)
            .map_err(|e| format!("{what}: header length: {e}"))?;
        let n = header_len(&what, len8, file_len)?;
        let mut header = vec![0u8; n as usize];
        f.read_exact(&mut header).map_err(|e| format!("{what}: header: {e}"))?;
        let data_start = 8 + n;
        let data_len = file_len - data_start;
        let (tensors, metadata) = parse_header(&header, data_len).map_err(|e| format!("{what}: {e}"))?;
        plain::check_payload_budget(&what, data_len, limit)?;
        Ok(Self {
            what,
            source: Source::File(f, stamp),
            data_start,
            data_len,
            limit,
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
            data_len: total - data_start,
            limit: None,
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

    /// Refuse, before it is made, an allocation of `elems` elements of
    /// `elem_size` bytes that exceeds the payload budget. `open` bounds the
    /// file's declared bytes; a read can allocate more than that (a 16-bit
    /// tensor widened to f32 takes twice the file's bytes), so each read
    /// checks its own allocation through the same owner.
    fn check_alloc(&self, name: &str, elems: usize, elem_size: usize) -> Result<(), String> {
        let bytes = (elems as u64).saturating_mul(elem_size as u64);
        plain::check_payload_budget(&format!("{}: {name}: read allocation", self.what), bytes, self.limit)
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
            Source::File(f, stamp) => {
                stamp.recheck(f, &self.what)?;
                f.read_exact_at(bytes, start)
                    .map_err(|e| format!("{name}: data: {e}"))?;
                stamp.recheck(f, &self.what)?;
            }
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
        self.check_alloc(name, info.numel(), std::mem::size_of::<T>())?;
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
        self.check_alloc(name, len, 1)?;
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
        self.check_alloc(name, info.numel(), 4)?;
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

/// What a model loader reads tensors through: one `.safetensors` file
/// ([`SafeTensors`]) or a sharded checkpoint ([`CheckpointView`]). The strict
/// per-name checks (dtype, shape, whole-tensor reads) live in the readers;
/// the loader above adds `refuse_unread` over [`Self::names`], so both apply
/// per name across every shard.
pub trait TensorSource {
    /// Every tensor name, sorted.
    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_>;
    /// See [`SafeTensors::read_f32`].
    fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String>;
    /// See [`SafeTensors::read_bf16_bits`].
    fn read_bf16_bits(&self, name: &str) -> Result<(Vec<usize>, Vec<u16>), String>;
    /// [`Self::read_f32`] into `dst`, after checking the dtype and `shape`.
    fn read_f32_into(&self, name: &str, shape: &[usize], dst: &mut [f32]) -> Result<(), String>;
    /// [`Self::read_bf16_bits`] into `dst`, after checking the dtype and `shape`.
    fn read_bf16_bits_into(&self, name: &str, shape: &[usize], dst: &mut [u16]) -> Result<(), String>;
}

impl TensorSource for SafeTensors {
    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(SafeTensors::names(self))
    }
    fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        SafeTensors::read_f32(self, name)
    }
    fn read_bf16_bits(&self, name: &str) -> Result<(Vec<usize>, Vec<u16>), String> {
        SafeTensors::read_bf16_bits(self, name)
    }
    fn read_f32_into(&self, name: &str, shape: &[usize], dst: &mut [f32]) -> Result<(), String> {
        SafeTensors::read_f32_into(self, name, shape, dst)
    }
    fn read_bf16_bits_into(&self, name: &str, shape: &[usize], dst: &mut [u16]) -> Result<(), String> {
        SafeTensors::read_bf16_bits_into(self, name, shape, dst)
    }
}

/// The index a sharded Hugging Face checkpoint ships beside its shards.
pub const INDEX_FILE_NAME: &str = "model.safetensors.index.json";

/// Largest index accepted: a 9B's is about 100 KB.
const MAX_INDEX_BYTES: u64 = 16 << 20;

/// The index's JSON: `{"metadata": {..}, "weight_map": {name: file}}`, two
/// container levels, integers only.
const INDEX_SYNTAX: Syntax = Syntax {
    what: "index JSON",
    max_depth: 2,
    uints_only: true,
    literals: false,
};

/// A checkpoint's tensors across one or several `.safetensors` files, each
/// name resolved through the index's `weight_map`.
///
/// [`Self::open`] validates everything before any tensor is read: the index
/// parses strictly; every shard is a bare file name in the checkpoint
/// directory (no `/`, `\`, `..` or NUL; a symlink is followed, so a Hugging
/// Face cache snapshot's `../../blobs` links load) that exists, is a regular
/// file and is a valid `.safetensors` file; every `weight_map` name is in the shard it
/// names; and every tensor in a shard is in the `weight_map` for that shard
/// (so a name present in two shards is refused, not shadowed). A single
/// `.safetensors` file opens as a one-shard view.
#[derive(Debug)]
pub struct CheckpointView {
    what: String,
    shards: Vec<SafeTensors>,
    /// Tensor name -> index into `shards`.
    owner: BTreeMap<String, usize>,
}

impl CheckpointView {
    /// Open `path`: a `.safetensors` file, a `model.safetensors.index.json`
    /// (or any `*.json` index), or a checkpoint directory holding an index
    /// or, failing that, `model.safetensors`.
    ///
    /// The payload budget defaults to this machine's physical memory
    /// (`sysconf(_SC_PHYS_PAGES) * _SC_PAGESIZE`) and applies to the shards'
    /// declared tensor bytes in total: a larger checkpoint is refused at open.
    /// To use another budget, call [`Self::open_with_limit`] with
    /// `Some(bytes)`.
    pub fn open(path: &Path) -> Result<Self, String> {
        Self::open_with_limit(path, None)
    }

    /// [`Self::open`], refusing, before any payload allocation, a checkpoint
    /// whose shards together declare more tensor bytes than `limit` (physical
    /// memory when `None`). The sum is checked as each shard opens, so an
    /// oversized shard is named before the rest are touched. The budget is
    /// computed by `plain::check_payload_budget`, the one owner shared with
    /// the `.npy` reader.
    pub fn open_with_limit(path: &Path, limit: Option<u64>) -> Result<Self, String> {
        let what = path.display().to_string();
        let md = std::fs::metadata(path).map_err(|e| format!("{what}: {e}"))?;
        if md.is_dir() {
            let index = path.join(INDEX_FILE_NAME);
            return match std::fs::symlink_metadata(&index) {
                Ok(_) => Self::open_index(&index, limit),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    Self::open_single(&path.join("model.safetensors"), limit)
                }
                Err(e) => Err(format!("{}: {e}", index.display())),
            };
        }
        if path.extension().is_some_and(|e| e == "json") {
            Self::open_index(path, limit)
        } else {
            Self::open_single(path, limit)
        }
    }

    /// Number of `.safetensors` files behind the view.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn open_single(path: &Path, limit: Option<u64>) -> Result<Self, String> {
        let st = SafeTensors::open_with_limit(path, limit)?;
        let owner = st.names().map(|n| (n.to_string(), 0)).collect();
        Ok(Self {
            what: path.display().to_string(),
            shards: vec![st],
            owner,
        })
    }

    fn open_index(index_path: &Path, limit: Option<u64>) -> Result<Self, String> {
        let what = index_path.display().to_string();
        let dir = match index_path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d,
            _ => Path::new("."),
        };
        let (f, len) = plain::open_regular_file(index_path)?;
        if len > MAX_INDEX_BYTES {
            return Err(format!(
                "{what}: {len} bytes, above the {MAX_INDEX_BYTES}-byte index cap"
            ));
        }
        let mut text = String::new();
        f.take(MAX_INDEX_BYTES)
            .read_to_string(&mut text)
            .map_err(|e| format!("{what}: {e}"))?;
        let by_file = parse_index(&text).map_err(|e| format!("{what}: {e}"))?;

        // Validate every shard name before opening any.
        let mut paths: Vec<(&String, std::path::PathBuf)> = Vec::with_capacity(by_file.len());
        for (file, names) in &by_file {
            paths.push((
                file,
                shard_path(dir, file, &names[0]).map_err(|e| format!("{what}: {e}"))?,
            ));
        }

        let mut shards = Vec::with_capacity(paths.len());
        let mut owner = BTreeMap::new();
        let mut declared = 0u64;
        for (i, ((file, path), names)) in paths.iter().zip(by_file.values()).enumerate() {
            let st = SafeTensors::open_with_limit(path, limit)?;
            declared = declared.saturating_add(st.data_len);
            plain::check_payload_budget(&what, declared, limit)?;
            for n in names {
                if st.info(n).is_err() {
                    return Err(format!(
                        "{what}: weight_map puts {n:?} in {file:?}, which has no such tensor"
                    ));
                }
            }
            for n in st.names() {
                if names.binary_search_by(|m| m.as_str().cmp(n)).is_err() {
                    let elsewhere = by_file.iter().find(|(_, ns)| ns.iter().any(|m| m == n));
                    return Err(match elsewhere {
                        Some((other, _)) => format!("{what}: tensor {n:?} is in both {file:?} and {other:?}"),
                        None => format!("{what}: {file:?} holds tensor {n:?}, which the weight_map does not list"),
                    });
                }
                owner.insert(n.to_string(), i);
            }
            shards.push(st);
        }
        Ok(Self { what, shards, owner })
    }

    /// `name`'s dtype and shape (and byte range within its shard), from the
    /// header alone: no tensor data is read. `None` if no shard has `name`.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.shard(name).ok()?.info(name).ok()
    }

    fn shard(&self, name: &str) -> Result<&SafeTensors, String> {
        match self.owner.get(name) {
            Some(&i) => Ok(&self.shards[i]),
            None => Err(format!("{}: no tensor {name:?}", self.what)),
        }
    }
}

impl TensorSource for CheckpointView {
    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.owner.keys().map(String::as_str))
    }
    fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let st = self.shard(name)?;
        st.read_f32(name)
    }
    fn read_bf16_bits(&self, name: &str) -> Result<(Vec<usize>, Vec<u16>), String> {
        self.shard(name)?.read_bf16_bits(name)
    }
    fn read_f32_into(&self, name: &str, shape: &[usize], dst: &mut [f32]) -> Result<(), String> {
        self.shard(name)?.read_f32_into(name, shape, dst)
    }
    fn read_bf16_bits_into(&self, name: &str, shape: &[usize], dst: &mut [u16]) -> Result<(), String> {
        self.shard(name)?.read_bf16_bits_into(name, shape, dst)
    }
}

/// The `weight_map` grouped by shard file: file -> its tensor names, sorted.
fn parse_index(text: &str) -> Result<BTreeMap<String, Vec<String>>, String> {
    let Json::Object(top) = json::parse(text, INDEX_SYNTAX)? else {
        return Err("index is not a JSON object".into());
    };
    let mut weight_map = None;
    for (k, v) in top {
        match (k.as_str(), v) {
            ("metadata", Json::Object(_)) => {}
            ("weight_map", Json::Object(entries)) => weight_map = Some(entries),
            (k, _) => return Err(format!("unexpected or mistyped field {k:?}")),
        }
    }
    let entries = weight_map.ok_or("no weight_map")?;
    if entries.is_empty() {
        return Err("weight_map is empty".into());
    }
    let mut by_file: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, file) in entries {
        let Json::Str(file) = file else {
            return Err(format!("weight_map entry {name:?} is not a file name"));
        };
        by_file.entry(file).or_default().push(name);
    }
    for names in by_file.values_mut() {
        names.sort_unstable();
    }
    Ok(by_file)
}

/// `dir/file`, for a `weight_map` value that is lexically a bare file name:
/// non-empty, no NUL, no `/` or `\\`, not `.` or `..` (so never absolute and
/// never a path out of `dir`). `tensor` names who asked for the file.
///
/// Symlinks are deliberately followed with no containment check: Hugging Face
/// snapshot directories hold every file as a symlink into the cache's
/// `../../blobs`. The index is the only untrusted input, and with a bare name
/// it cannot steer the path anywhere but an entry of the checkpoint directory;
/// what such an entry is (a FIFO, a directory) is decided by
/// [`plain::open_regular_file`] on the handle that is read.
fn shard_path(dir: &Path, file: &str, tensor: &str) -> Result<std::path::PathBuf, String> {
    let bare = !file.is_empty() && !file.contains(['/', '\\', '\0']) && file != "." && file != "..";
    if !bare {
        return Err(format!(
            "shard {file:?} (for {tensor:?}) is not a bare file name in the checkpoint directory"
        ));
    }
    Ok(dir.join(file))
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
        let numel = plain::checked_numel(&t.shape).ok_or_else(|| format!("{name}: element count overflows"))? as u64;
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

#[cfg(test)]
mod tests {
    use super::widen_in_place;
    use crate::tensor::{bf16_bits_to_f32, f16_bits_to_f32};

    /// The blocked in-place widen gives the same bytes as widening element by
    /// element into a separate buffer, at sizes below, at and around the
    /// block threshold (`2 * 1024`) and with a ragged tail.
    #[test]
    fn widen_in_place_matches_elementwise() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for &n in &[0usize, 1, 2, 1023, 2047, 2048, 2049, 3072, 4096, 5000, 10_001] {
            let src: Vec<u16> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (seed >> 48) as u16
                })
                .collect();
            for (name, widen) in [("bf16", bf16_bits_to_f32 as fn(u16) -> f32), ("f16", f16_bits_to_f32)] {
                let mut bytes = vec![0u8; 4 * n];
                for (i, v) in src.iter().enumerate() {
                    bytes[2 * n + 2 * i..2 * n + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
                }
                widen_in_place(&mut bytes, n, widen);
                let want: Vec<u8> = src.iter().flat_map(|&v| widen(v).to_ne_bytes()).collect();
                assert_eq!(bytes, want, "{name}, n = {n}");
            }
        }
    }
}
