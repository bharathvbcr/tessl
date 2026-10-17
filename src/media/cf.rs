//! CoreFoundation, the part of it the media decoders share: owned references
//! and the few calls that build a `CFData` or `CFURL` from Rust bytes.
//!
//! Declared by hand from `CoreFoundation/CFBase.h`, `CFData.h`, `CFURL.h`,
//! `CFNumber.h`, `CFDictionary.h` and `CFString.h` of the macOS SDK. Every
//! Core Foundation, Core Graphics, ImageIO, CoreMedia and AudioToolbox object
//! this crate creates is held by a [`CfRef`], so a `?` on any path releases it.

use std::ffi::c_void;
use std::ptr::NonNull;

/// `CFTypeRef`: any Core Foundation object, as an untyped pointer.
pub type CfTypeRef = *const c_void;
/// `CFIndex` (`long`).
pub type CfIndex = isize;
/// `Boolean` (`unsigned char`).
pub type Boolean = u8;

/// `kCFStringEncodingUTF8` (`CFStringEncodings`, `CFString.h`).
pub const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
/// `kCFNumberSInt32Type` (`CFNumber.h`).
pub const K_CF_NUMBER_SINT32_TYPE: CfIndex = 3;
/// `kCFNumberSInt64Type` (`CFNumber.h`).
pub const K_CF_NUMBER_SINT64_TYPE: CfIndex = 4;

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    pub fn CFRelease(cf: CfTypeRef);
    pub fn CFEqual(a: CfTypeRef, b: CfTypeRef) -> Boolean;
    pub fn CFGetTypeID(cf: CfTypeRef) -> usize;
    pub fn CFNumberGetTypeID() -> usize;
    pub fn CFStringGetTypeID() -> usize;
    pub fn CFDataCreate(allocator: CfTypeRef, bytes: *const u8, length: CfIndex) -> CfTypeRef;
    pub fn CFDataGetLength(data: CfTypeRef) -> CfIndex;
    pub fn CFDataGetBytePtr(data: CfTypeRef) -> *const u8;
    pub fn CFURLCreateFromFileSystemRepresentation(
        allocator: CfTypeRef,
        buffer: *const u8,
        buf_len: CfIndex,
        is_directory: Boolean,
    ) -> CfTypeRef;
    pub fn CFDictionaryGetValue(dict: CfTypeRef, key: *const c_void) -> *const c_void;
    pub fn CFNumberGetValue(number: CfTypeRef, the_type: CfIndex, value_ptr: *mut c_void) -> Boolean;
    pub fn CFStringGetCString(string: CfTypeRef, buffer: *mut u8, buffer_size: CfIndex, encoding: u32) -> Boolean;
}

/// An owned Core Foundation reference (a "Create" or "Copy" result, +1),
/// released exactly once when dropped. `CFRelease` is the release for every
/// CF-based type, `CGImageRef`, `CGColorSpaceRef`, `CGContextRef`,
/// `CGDataProviderRef`, `CMSampleBufferRef` and `CGImageSourceRef` included.
#[derive(Debug)]
pub struct CfRef(NonNull<c_void>);

impl CfRef {
    /// Take ownership of a +1 reference; `None` for NULL.
    ///
    /// # Safety
    /// `ptr` must be NULL or a live Core Foundation object the caller owns one
    /// reference to (a `Create`/`Copy` result), and must not be released
    /// elsewhere.
    pub unsafe fn take(ptr: CfTypeRef) -> Option<CfRef> {
        NonNull::new(ptr as *mut c_void).map(CfRef)
    }

    /// The raw reference, valid while `self` lives.
    pub fn as_ptr(&self) -> CfTypeRef {
        self.0.as_ptr() as CfTypeRef
    }
}

impl Drop for CfRef {
    fn drop(&mut self) {
        // SAFETY: `self` owns one reference (see `take`), and `drop` runs once.
        unsafe { CFRelease(self.as_ptr()) }
    }
}

/// A `CFData` holding a copy of `bytes`.
pub fn data_from_bytes(bytes: &[u8]) -> Option<CfRef> {
    let len = CfIndex::try_from(bytes.len()).ok()?;
    // SAFETY: the pointer and length describe `bytes`; a NULL allocator is the
    // default one; CFDataCreate copies the bytes and returns +1.
    unsafe { CfRef::take(CFDataCreate(std::ptr::null(), bytes.as_ptr(), len)) }
}

/// The bytes of a `CFData`, valid while `data` lives.
///
/// # Safety
/// `data` must be a live `CFData`.
pub unsafe fn data_bytes(data: &CfRef) -> &[u8] {
    let len = CFDataGetLength(data.as_ptr());
    let ptr = CFDataGetBytePtr(data.as_ptr());
    if ptr.is_null() || len <= 0 {
        return &[];
    }
    // SAFETY (caller): `data` is a live CFData; CFDataGetBytePtr points at
    // `CFDataGetLength` bytes that stay valid while `data` is not mutated or
    // released, and the returned slice borrows `data`.
    std::slice::from_raw_parts(ptr, len as usize)
}

/// A `CFURL` for a file system path, built from the path's own bytes (so a
/// path that is not UTF-8 still works).
pub fn file_url(path: &std::path::Path) -> Option<CfRef> {
    use std::os::unix::ffi::OsStrExt;
    let bytes = path.as_os_str().as_bytes();
    let len = CfIndex::try_from(bytes.len()).ok()?;
    // SAFETY: pointer and length describe `bytes`; the result is +1 or NULL.
    unsafe {
        CfRef::take(CFURLCreateFromFileSystemRepresentation(
            std::ptr::null(),
            bytes.as_ptr(),
            len,
            0,
        ))
    }
}

/// An `i64` from a `CFNumber` found in `dict` under `key`; `None` when the key
/// is absent or not a number.
///
/// # Safety
/// `dict` must be a live `CFDictionary` and `key` a live CF string constant.
pub unsafe fn dict_i64(dict: CfTypeRef, key: CfTypeRef) -> Option<i64> {
    let value = CFDictionaryGetValue(dict, key);
    if value.is_null() || CFGetTypeID(value) != CFNumberGetTypeID() {
        return None;
    }
    let mut out: i64 = 0;
    // SAFETY: `value` is a CFNumber (checked) and `out` is an i64, the size
    // kCFNumberSInt64Type names.
    if CFNumberGetValue(value, K_CF_NUMBER_SINT64_TYPE, (&mut out as *mut i64).cast()) != 0 {
        Some(out)
    } else {
        None
    }
}

/// A `CFString` as a Rust string (lossy only if the string is not
/// representable, which UTF-8 always is); `None` for NULL or a non-string.
///
/// # Safety
/// `s` must be NULL or a live Core Foundation object.
pub unsafe fn cf_string(s: CfTypeRef) -> Option<String> {
    if s.is_null() || CFGetTypeID(s) != CFStringGetTypeID() {
        return None;
    }
    let mut buf = [0u8; 256];
    // SAFETY: `s` is a CFString (checked); the buffer length is passed.
    if CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as CfIndex, K_CF_STRING_ENCODING_UTF8) == 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0)?;
    Some(String::from_utf8_lossy(&buf[..end]).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_round_trips_and_releases() {
        let d = data_from_bytes(&[1, 2, 3, 4, 5]).expect("CFData");
        // SAFETY: `d` is a live CFData.
        assert_eq!(unsafe { data_bytes(&d) }, &[1, 2, 3, 4, 5]);
        let empty = data_from_bytes(&[]).expect("empty CFData");
        // SAFETY: `empty` is a live CFData.
        assert!(unsafe { data_bytes(&empty) }.is_empty());
    }

    #[test]
    fn a_path_that_is_not_utf8_still_makes_a_url() {
        use std::os::unix::ffi::OsStrExt;
        let p = std::path::Path::new(std::ffi::OsStr::from_bytes(b"/tmp/\xff\xfe.wav"));
        assert!(file_url(p).is_some());
    }
}
