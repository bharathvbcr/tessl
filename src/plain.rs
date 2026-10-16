//! Filling typed slices straight from little-endian file bytes.
//!
//! The checkpoint readers (`.npy`, `.safetensors`) read multi-gigabyte
//! payloads. Reading them into a byte buffer and then converting element by
//! element holds the tensor twice; filling the destination's own storage
//! holds it once. This module owns the one `unsafe` view that makes that
//! possible, so each reader does not keep its own.

/// Scalar types for which every bit pattern is a valid value, so filling one
/// from raw file bytes cannot produce an invalid inhabitant.
///
/// # Safety
///
/// Implementors must have no invalid bit patterns and no padding. `bool` and
/// `char` must never implement this; the integer and IEEE-754 float types may.
pub(crate) unsafe trait PlainScalar: Copy {
    /// The value whose in-memory bytes are `self`'s bytes read as
    /// little-endian: the identity on a little-endian host.
    #[cfg(target_endian = "big")]
    fn from_le(self) -> Self;
}

macro_rules! plain_int {
    ($($t:ty),*) => {$(
        // SAFETY: a primitive integer: no padding, every bit pattern valid.
        unsafe impl PlainScalar for $t {
            #[cfg(target_endian = "big")]
            fn from_le(self) -> Self {
                <$t>::from_le(self)
            }
        }
    )*};
}
plain_int!(u8, i8, u16, i16, u32, i32, u64, i64);

macro_rules! plain_float {
    ($($t:ty),*) => {$(
        // SAFETY: an IEEE-754 float: no padding, every bit pattern is a value
        // (NaN payloads included).
        unsafe impl PlainScalar for $t {
            #[cfg(target_endian = "big")]
            fn from_le(self) -> Self {
                <$t>::from_bits(self.to_bits().swap_bytes())
            }
        }
    )*};
}
plain_float!(f32, f64);

/// `dst`'s storage as bytes, to be filled by a read.
pub(crate) fn bytes_mut<T: PlainScalar>(dst: &mut [T]) -> &mut [u8] {
    // SAFETY: reinterprets a uniquely borrowed slice's storage as the bytes a
    // read fills. The pointer is valid and exclusively borrowed for the
    // returned lifetime, the length is `size_of_val` of that same slice so it
    // cannot overrun, `u8` has no alignment requirement, and `PlainScalar`
    // guarantees any bytes written leave every element a valid `T` — the file
    // may hold nonsense numbers but never an invalid value.
    unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(dst)) }
}

/// Turn elements just filled from little-endian bytes into native values: a
/// no-op on a little-endian host.
pub(crate) fn le_to_native<T: PlainScalar>(dst: &mut [T]) {
    #[cfg(target_endian = "big")]
    for x in dst.iter_mut() {
        *x = x.from_le();
    }
    #[cfg(target_endian = "little")]
    let _ = dst;
}

/// Open `path` for a checkpoint reader, refusing anything but a regular file.
///
/// `File::open` on a FIFO with no writer blocks forever, and a character
/// device or directory has no meaningful length to validate a header against.
/// The open is `O_NONBLOCK`, which returns at once for a FIFO, and the handle
/// is then fstat-ed and must be a regular file; the check is on the handle that
/// is read, so no path swap can slip a FIFO past it. There is deliberately no
/// stat before the open: it would add nothing the fstat does not decide.
/// `O_NONBLOCK` has no effect on reads of a regular file, which is all that is
/// kept. Returns the handle and its length.
pub(crate) fn open_regular_file(path: &std::path::Path) -> Result<(std::fs::File, u64), String> {
    use std::os::unix::fs::OpenOptionsExt;
    let what = path.display();
    let f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
        .map_err(|e| format!("{what}: {e}"))?;
    let md = f.metadata().map_err(|e| format!("{what}: {e}"))?;
    if !md.is_file() {
        return Err(format!("{what}: not a regular file"));
    }
    Ok((f, md.len()))
}

/// The element count of `shape`, or `None` if it does not fit in `usize`.
///
/// A zero dimension makes the count zero whatever the other dimensions are, so
/// the verdict does not depend on the order of the dims: a left-to-right
/// `checked_mul` fold would collapse `[0, 2^32, 2^32]` to 0 before reaching the
/// overflow and refuse `[2^32, 2^32, 0]`, though both name the same empty
/// tensor. The one owner of this product for the `.npy` and `.safetensors`
/// readers.
pub(crate) fn checked_numel(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// This machine's physical memory in bytes: the default payload budget of the
/// CPU-side checkpoint readers, which hold a tensor in host memory.
fn physical_memory_bytes() -> Result<u64, String> {
    // SAFETY: `sysconf` takes an integer name and has no memory preconditions;
    // a failure is reported as -1.
    let (pages, page) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
    match (u64::try_from(pages), u64::try_from(page)) {
        (Ok(pages), Ok(page)) if pages > 0 && page > 0 => pages
            .checked_mul(page)
            .ok_or_else(|| "physical memory size overflows u64".to_string()),
        _ => Err("cannot determine physical memory for the default payload budget".to_string()),
    }
}

/// Refuse, before any payload allocation, a checkpoint whose declared tensor
/// bytes exceed the budget: `limit` when the caller gave one, else
/// [`physical_memory_bytes`]. The one place that budget is computed and
/// compared, so no reader keeps its own check. A budget that cannot be
/// determined is an error, never "no limit".
pub(crate) fn check_payload_budget(what: &str, declared: u64, limit: Option<u64>) -> Result<(), String> {
    let (budget, source) = match limit {
        Some(l) => (l, "the caller's limit"),
        None => (
            physical_memory_bytes().map_err(|e| format!("{what}: {e}"))?,
            "physical memory",
        ),
    };
    if declared > budget {
        return Err(format!(
            "{what}: payload budget exceeded: the file declares {declared} tensor bytes, above {source} of {budget} bytes"
        ));
    }
    Ok(())
}
