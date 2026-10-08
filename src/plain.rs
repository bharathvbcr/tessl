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
