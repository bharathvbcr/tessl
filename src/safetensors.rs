//! tessl's view of the `.safetensors` checkpoint format: a thin adapter over
//! ojas-io's reader.
//!
//! The owner decided on 2026-11-01 that **ojas owns IO**. `ojas_io::safetensors`
//! is the canonical reader — the header grammar, the schema walk, the tiling of
//! the data buffer, dtype widths, positioned reads, the file-identity re-check
//! and the index bound all live there, once, for both repositories. This module
//! keeps only what is tessl's:
//!
//! * **tessl's surface.** `String` errors, `Vec<usize>` shapes, and the method
//!   names its loaders and tests already call, so none of the 47 files outside
//!   this one had to change.
//! * **tessl's dtype contract.** [`Dtype`] has nineteen variants and no
//!   sub-byte float. ojas-io supports `F4`, `F6_E2M3` and `F6_E3M2` on a
//!   checked path (it sizes them from `bits()` and refuses a span that is not
//!   bit-exact); tessl declines them, because a whole-byte range cannot be
//!   stated for them and every tessl path downstream of here is written in
//!   bytes. [`TryFrom<StDtype>`] is where that refusal lives, so it is enforced
//!   by the type at open rather than documented and hoped for.
//! * **tessl's payload policy.** A `None` limit means this machine's physical
//!   memory here, and a checkpoint declaring more is refused at open. ojas-io
//!   treats `None` as unbounded on purpose — it validates a header without
//!   loading the file and reads positioned from disk, so opening a checkpoint
//!   larger than memory is legitimate there — but tessl's readers hand whole
//!   tensors to host code, so a pre-check fits them. The budget, its wording
//!   and the per-read allocation check are all computed by
//!   [`plain::check_payload_budget`], still the one owner shared with the
//!   `.npy` reader.
//!
//! What this module no longer contains is the parser: about 950 lines of
//! header grammar, offset validation and positioned reads that were a second
//! implementation of ojas-io's. Everything that file refused is still refused;
//! `ojas-io/tests/tessl_rejection_corpus.rs` holds that corpus against the
//! reader underneath, and the cases where the two word a refusal differently
//! are listed in the commit that made this change.

use std::collections::BTreeMap;
use std::path::Path;

use ojas_io::{StDtype, TensorSource as IoTensorSource};

use crate::plain;

/// Largest header accepted. The format's reference implementation caps it at
/// 100 MB; a 2B checkpoint's header is about 90 KB. The cap itself is
/// ojas-io's; this is the name tessl's callers use for it.
pub const MAX_HEADER_BYTES: u64 = ojas_io::MAX_HEADER_BYTES;

/// Element types the format defines in whole bytes. [`Dtype::F32`],
/// [`Dtype::F16`] and [`Dtype::BF16`] read back as f32
/// ([`SafeTensors::read_f32`]); [`Dtype::U8`], [`Dtype::I8`] and
/// [`Dtype::U32`] (int8 and MLX-packed Q4 weights) as integers; every dtype,
/// fp8 included, as its raw bytes ([`SafeTensors::read_raw`]). The format's
/// sub-byte types (`F4`, `F6_E2M3`, `F6_E3M2`) are refused: their sizes are
/// not whole bytes, so the byte ranges every tessl path below here is written
/// in cannot be stated for them.
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

impl TryFrom<StDtype> for Dtype {
    type Error = String;

    /// Where tessl's refusal of the sub-byte floats lives.
    ///
    /// ojas-io's reader has three variants tessl does not, and it accepts them
    /// on a bit-exact span. A conversion that cannot fail would put them in
    /// tessl's namespace, break every exhaustive `match` on [`Dtype`], and hand
    /// byte-oriented code a dtype whose size is not a whole number of bytes.
    /// Erroring here means the refusal happens once, at open, for every route
    /// into this module.
    fn try_from(d: StDtype) -> Result<Self, String> {
        Ok(match d {
            StDtype::Bool => Dtype::Bool,
            StDtype::U8 => Dtype::U8,
            StDtype::I8 => Dtype::I8,
            StDtype::F8E4M3 => Dtype::F8E4M3,
            StDtype::F8E4M3Fnuz => Dtype::F8E4M3Fnuz,
            StDtype::F8E5M2 => Dtype::F8E5M2,
            StDtype::F8E5M2Fnuz => Dtype::F8E5M2Fnuz,
            StDtype::F8E8M0 => Dtype::F8E8M0,
            StDtype::I16 => Dtype::I16,
            StDtype::U16 => Dtype::U16,
            StDtype::F16 => Dtype::F16,
            StDtype::BF16 => Dtype::BF16,
            StDtype::I32 => Dtype::I32,
            StDtype::U32 => Dtype::U32,
            StDtype::F32 => Dtype::F32,
            StDtype::F64 => Dtype::F64,
            StDtype::I64 => Dtype::I64,
            StDtype::U64 => Dtype::U64,
            StDtype::C64 => Dtype::C64,
            StDtype::F4 => return Err("unsupported dtype \"F4\"".into()),
            StDtype::F6E2M3 => return Err("unsupported dtype \"F6_E2M3\"".into()),
            StDtype::F6E3M2 => return Err("unsupported dtype \"F6_E3M2\"".into()),
        })
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

/// Every tensor's entry, in tessl's terms, from an open ojas-io reader.
///
/// This is where the dtype conversion and the `u64` to `usize` shape narrowing
/// happen: once, at open, for the whole header, so no read path can meet a
/// dtype tessl does not have or a dimension that does not fit this host.
fn entries<S: IoTensorSource>(
    source: &S,
    what: &str,
    info: impl Fn(&S, &str) -> Option<(StDtype, Vec<u64>, u64, u64)>,
) -> Result<BTreeMap<String, TensorInfo>, String> {
    let names: Vec<String> = source.names().map(|n| n.to_string()).collect();
    let mut out = BTreeMap::new();
    for name in names {
        let (dtype, shape, begin, end) = info(source, &name)
            .ok_or_else(|| format!("{what}: {name:?} is listed but has no header entry"))?;
        let dtype = Dtype::try_from(dtype).map_err(|e| format!("{what}: {name}: {e}"))?;
        let mut dims = Vec::with_capacity(shape.len());
        for d in shape {
            dims.push(
                usize::try_from(d)
                    .map_err(|_| format!("{what}: {name}: dimension {d} overflows usize"))?,
            );
        }
        if plain::checked_numel(&dims).is_none() {
            return Err(format!("{what}: {name}: element count overflows"));
        }
        out.insert(
            name,
            TensorInfo {
                dtype,
                shape: dims,
                begin,
                end,
            },
        );
    }
    Ok(out)
}

/// An open `.safetensors` file (or its bytes): its parsed header and where its
/// data starts.
#[derive(Debug)]
pub struct SafeTensors {
    /// The path, or the label [`SafeTensors::from_bytes`] was given: what
    /// errors name.
    what: String,
    inner: ojas_io::SafeTensors<'static>,
    tensors: BTreeMap<String, TensorInfo>,
    /// The caller's payload limit (`None`: physical memory), applied again to
    /// each read's own allocation.
    limit: Option<u64>,
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
        // ojas-io is given no limit: the budget below is tessl's own policy,
        // including what `None` means, and it is stated in tessl's words.
        let inner = ojas_io::SafeTensors::open(path).map_err(|e| e.detail().to_string())?;
        Self::finish(what, inner, limit)
    }

    /// Validate `bytes`, a whole `.safetensors` serialization held in memory,
    /// exactly as [`Self::open`] validates a file; errors name `what`. For
    /// checkpoints built on the host (tests, conversions) that never need to
    /// touch the file system.
    pub fn from_bytes(what: &str, bytes: Vec<u8>) -> Result<Self, String> {
        let inner = ojas_io::SafeTensors::from_bytes(bytes)
            .map_err(|e| format!("{what}: {}", e.detail()))?;
        Self::finish(what.to_string(), inner, None)
    }

    fn finish(
        what: String,
        inner: ojas_io::SafeTensors<'static>,
        limit: Option<u64>,
    ) -> Result<Self, String> {
        let tensors = entries(&inner, &what, |s, n| {
            s.info(n)
                .ok()
                .map(|i| (i.dtype, i.shape.clone(), i.begin, i.end))
        })?;
        plain::check_payload_budget(&what, inner.declared_payload_bytes(), limit)?;
        Ok(Self {
            what,
            inner,
            tensors,
            limit,
        })
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The `__metadata__` entries, if any.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        self.inner.metadata()
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
        plain::check_payload_budget(
            &format!("{}: {name}: read allocation", self.what),
            bytes,
            self.limit,
        )
    }

    /// A tensor's header entry and its bytes exactly as stored (little-endian
    /// elements), whatever its dtype: the way to read fp8, or any type this
    /// module has no typed reader for.
    pub fn read_raw(&self, name: &str) -> Result<(TensorInfo, Vec<u8>), String> {
        let info = self.info(name)?.clone();
        let len = usize::try_from(info.end - info.begin)
            .map_err(|_| format!("{name}: tensor too large for this host"))?;
        self.check_alloc(name, len, 1)?;
        let bytes = self
            .inner
            .read_bytes(name)
            .map_err(|e| e.detail().to_string())?;
        Ok((info, bytes))
    }

    /// A U8 tensor and its shape. Any other dtype is an error.
    pub fn read_u8(&self, name: &str) -> Result<(Vec<usize>, Vec<u8>), String> {
        self.check_alloc(name, self.info(name)?.numel(), 1)?;
        shaped(self.inner.read_u8(name))
    }

    /// An I8 tensor (int8 weights) and its shape. Any other dtype is an error.
    pub fn read_i8(&self, name: &str) -> Result<(Vec<usize>, Vec<i8>), String> {
        self.check_alloc(name, self.info(name)?.numel(), 1)?;
        shaped(self.inner.read_i8(name))
    }

    /// A U32 tensor (MLX packs eight 4-bit weights per element) and its shape.
    /// Any other dtype is an error.
    pub fn read_u32(&self, name: &str) -> Result<(Vec<usize>, Vec<u32>), String> {
        self.check_alloc(name, self.info(name)?.numel(), 4)?;
        shaped(self.inner.read_u32(name))
    }

    /// A bf16 tensor's raw bit patterns and shape. Any other dtype is an error:
    /// narrowing is the caller's decision, not the loader's.
    pub fn read_bf16_bits(&self, name: &str) -> Result<(Vec<usize>, Vec<u16>), String> {
        self.check_alloc(name, self.info(name)?.numel(), 2)?;
        shaped(self.inner.read_bf16_bits(name))
    }

    /// [`Self::read_bf16_bits`] into `dst` (a device tensor's shared storage,
    /// say) instead of a new `Vec`, after checking the dtype and `shape`.
    pub(crate) fn read_bf16_bits_into(
        &self,
        name: &str,
        shape: &[usize],
        dst: &mut [u16],
    ) -> Result<(), String> {
        self.inner
            .read_bf16_bits_into(name, shape, dst)
            .map_err(|e| e.detail().to_string())
    }

    /// [`Self::read_u32`] into `dst` (a device buffer's shared storage, say)
    /// after checking the dtype and `shape`.
    pub(crate) fn read_u32_into(
        &self,
        name: &str,
        shape: &[usize],
        dst: &mut [u32],
    ) -> Result<(), String> {
        self.inner
            .read_u32_into(name, shape, dst)
            .map_err(|e| e.detail().to_string())
    }

    /// A floating tensor widened exactly to f32 (F32, BF16 or F16; every value
    /// of the narrower types is representable in f32).
    pub fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        self.check_alloc(name, self.info(name)?.numel(), 4)?;
        IoTensorSource::read_f32(&self.inner, name).map_err(|e| e.detail().to_string())
    }

    /// [`Self::read_f32`] into `dst` (a device tensor's shared storage, say)
    /// instead of a new `Vec`, after checking the dtype and `shape`.
    pub(crate) fn read_f32_into(
        &self,
        name: &str,
        shape: &[usize],
        dst: &mut [f32],
    ) -> Result<(), String> {
        self.inner
            .read_f32_into(name, shape, dst)
            .map_err(|e| e.detail().to_string())
    }
}

/// An ojas-io read's `(Vec<u64>, values)` in tessl's `Vec<usize>` shape.
/// The dimensions were narrowed and checked at open, so this cannot fail.
fn shaped<T>(
    r: Result<(Vec<u64>, Vec<T>), ojas_io::IoError>,
) -> Result<(Vec<usize>, Vec<T>), String> {
    let (shape, values) = r.map_err(|e| e.detail().to_string())?;
    Ok((shape.into_iter().map(|d| d as usize).collect(), values))
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
    fn read_bf16_bits_into(
        &self,
        name: &str,
        shape: &[usize],
        dst: &mut [u16],
    ) -> Result<(), String>;
    /// `name`'s dtype and shape, from the header alone (no tensor data is read).
    fn tensor_info(&self, name: &str) -> Result<TensorInfo, String>;
    /// A U32 tensor (MLX packs eight 4-bit weights per element) into `dst`,
    /// after checking the dtype and `shape`.
    fn read_u32_into(&self, name: &str, shape: &[usize], dst: &mut [u32]) -> Result<(), String>;
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
    fn read_bf16_bits_into(
        &self,
        name: &str,
        shape: &[usize],
        dst: &mut [u16],
    ) -> Result<(), String> {
        SafeTensors::read_bf16_bits_into(self, name, shape, dst)
    }
    fn tensor_info(&self, name: &str) -> Result<TensorInfo, String> {
        self.info(name).cloned()
    }
    fn read_u32_into(&self, name: &str, shape: &[usize], dst: &mut [u32]) -> Result<(), String> {
        SafeTensors::read_u32_into(self, name, shape, dst)
    }
}

/// The index a sharded Hugging Face checkpoint ships beside its shards.
pub const INDEX_FILE_NAME: &str = ojas_io::INDEX_FILE_NAME;

/// A checkpoint's tensors across one or several `.safetensors` files, each
/// name resolved through the index's `weight_map`.
///
/// [`Self::open`] validates everything before any tensor is read: the index
/// parses strictly and is bounded before it is allocated; every shard is a
/// bare file name in the checkpoint directory (no `/`, `\`, `..` or NUL; a
/// symlink is followed, so a Hugging Face cache snapshot's `../../blobs` links
/// load) that exists, is a regular file and is a valid `.safetensors` file;
/// every `weight_map` name is in the shard it names; and every tensor in a
/// shard is in the `weight_map` for that shard (so a name present in two
/// shards is refused, not shadowed). A single `.safetensors` file opens as a
/// one-shard view. All of that is ojas-io's; what this adds is tessl's dtype
/// contract and tessl's payload budget over the checkpoint as a whole.
#[derive(Debug)]
pub struct CheckpointView {
    what: String,
    inner: ojas_io::CheckpointView,
    tensors: BTreeMap<String, TensorInfo>,
    limit: Option<u64>,
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
    /// memory when `None`). The budget is computed by
    /// `plain::check_payload_budget`, the one owner shared with the `.npy`
    /// reader.
    pub fn open_with_limit(path: &Path, limit: Option<u64>) -> Result<Self, String> {
        let what = path.display().to_string();
        let inner = ojas_io::CheckpointView::open(path).map_err(|e| e.detail().to_string())?;
        let tensors = entries(&inner, &what, |s, n| {
            s.tensor_info(n)
                .map(|i| (i.dtype, i.shape.clone(), i.begin, i.end))
        })?;
        plain::check_payload_budget(&what, inner.declared_payload_bytes(), limit)?;
        Ok(Self {
            what,
            inner,
            tensors,
            limit,
        })
    }

    /// Number of `.safetensors` files behind the view.
    pub fn shard_count(&self) -> usize {
        self.inner.shard_count()
    }

    /// `name`'s dtype and shape (and byte range within its shard), from the
    /// header alone: no tensor data is read. `None` if no shard has `name`.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    fn entry(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("{}: no tensor {name:?}", self.what))
    }

    fn check_alloc(&self, name: &str, elems: usize, elem_size: usize) -> Result<(), String> {
        let bytes = (elems as u64).saturating_mul(elem_size as u64);
        plain::check_payload_budget(
            &format!("{}: {name}: read allocation", self.what),
            bytes,
            self.limit,
        )
    }
}

impl TensorSource for CheckpointView {
    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.tensors.keys().map(String::as_str))
    }
    fn read_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        self.check_alloc(name, self.entry(name)?.numel(), 4)?;
        IoTensorSource::read_f32(&self.inner, name).map_err(|e| e.detail().to_string())
    }
    fn read_bf16_bits(&self, name: &str) -> Result<(Vec<usize>, Vec<u16>), String> {
        self.check_alloc(name, self.entry(name)?.numel(), 2)?;
        IoTensorSource::read_bf16_bits(&self.inner, name).map_err(|e| e.detail().to_string())
    }
    fn read_f32_into(&self, name: &str, shape: &[usize], dst: &mut [f32]) -> Result<(), String> {
        IoTensorSource::read_f32_into(&self.inner, name, shape, dst)
            .map_err(|e| e.detail().to_string())
    }
    fn read_bf16_bits_into(
        &self,
        name: &str,
        shape: &[usize],
        dst: &mut [u16],
    ) -> Result<(), String> {
        IoTensorSource::read_bf16_bits_into(&self.inner, name, shape, dst)
            .map_err(|e| e.detail().to_string())
    }
    fn tensor_info(&self, name: &str) -> Result<TensorInfo, String> {
        self.entry(name).cloned()
    }
    fn read_u32_into(&self, name: &str, shape: &[usize], dst: &mut [u32]) -> Result<(), String> {
        IoTensorSource::read_u32_into(&self.inner, name, shape, dst)
            .map_err(|e| e.detail().to_string())
    }
}
