//! Minimal NumPy `.npy` reader (v1.0 / v2.0, C-order, f32 / i64).

use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

/// Defensive ceiling for caller-controlled metadata. A normal NPY header is a
/// few hundred bytes; 16 MiB still permits millions of dimensions while
/// preventing a four-byte v2 length field from turning a tiny hostile file
/// into a multi-gigabyte allocation attempt.
const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data_f32: Option<Vec<f32>>,
    pub data_i64: Option<Vec<i64>>,
}

impl NpyArray {
    pub fn f32_slice(&self) -> Result<&[f32], String> {
        self.data_f32
            .as_deref()
            .ok_or_else(|| "expected float32 npy".into())
    }

    pub fn i64_slice(&self) -> Result<&[i64], String> {
        self.data_i64
            .as_deref()
            .ok_or_else(|| "expected int64 npy".into())
    }

    pub fn scalar_f32(&self) -> Result<f32, String> {
        let s = self.f32_slice()?;
        if s.len() != 1 {
            return Err(format!("expected scalar, got {} elems", s.len()));
        }
        Ok(s[0])
    }
}

fn checked_shape_numel(shape: &[usize], operation: &str) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| format!("{operation}: shape element count overflow"))
}

fn checked_payload_nbytes(
    numel: usize,
    element_size: usize,
    operation: &str,
) -> Result<usize, String> {
    numel
        .checked_mul(element_size)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| format!("{operation}: payload byte size overflow"))
}

fn zeroed_payload<T: Clone + Default>(numel: usize, operation: &str) -> Result<Vec<T>, String> {
    let mut data = Vec::new();
    data.try_reserve_exact(numel)
        .map_err(|e| format!("{operation}: payload allocation failed: {e}"))?;
    data.resize(numel, T::default());
    Ok(data)
}

fn ensure_bytes_available(
    file: &File,
    start: u64,
    expected_bytes: usize,
    what: &str,
) -> Result<(), String> {
    let file_len = file
        .metadata()
        .map_err(|e| format!("read_npy: {what} metadata: {e}"))?
        .len();
    let remaining = file_len
        .checked_sub(start)
        .ok_or_else(|| format!("read_npy: {what} offset exceeds file length"))?;
    let expected = u64::try_from(expected_bytes)
        .map_err(|_| format!("read_npy: {what} byte size does not fit the file format"))?;
    if remaining < expected {
        return Err(format!(
            "read_npy: {what} needs {expected} bytes, but only {remaining} remain"
        ));
    }
    Ok(())
}

pub fn read_npy(path: &Path) -> Result<NpyArray, String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut magic = [0u8; 6];
    f.read_exact(&mut magic)
        .map_err(|e| format!("read magic: {e}"))?;
    if &magic != b"\x93NUMPY" {
        return Err(format!("not an npy file: {}", path.display()));
    }
    let mut ver = [0u8; 2];
    f.read_exact(&mut ver).map_err(|e| format!("ver: {e}"))?;
    let header_len: usize = match ver {
        [1, 0] => {
            let mut hl = [0u8; 2];
            f.read_exact(&mut hl).map_err(|e| format!("hlen: {e}"))?;
            usize::from(u16::from_le_bytes(hl))
        }
        [2, 0] => {
            let mut hl = [0u8; 4];
            f.read_exact(&mut hl).map_err(|e| format!("hlen: {e}"))?;
            usize::try_from(u32::from_le_bytes(hl))
                .map_err(|_| "read_npy: v2 header length does not fit usize".to_string())?
        }
        [major, minor] => {
            return Err(format!(
                "unsupported npy version {major}.{minor}; expected 1.0 or 2.0"
            ))
        }
    };
    if header_len > MAX_HEADER_BYTES {
        return Err(format!(
            "read_npy: header length {header_len} exceeds {MAX_HEADER_BYTES}-byte safety bound"
        ));
    }
    let header_start = f
        .stream_position()
        .map_err(|e| format!("read_npy: header position: {e}"))?;
    ensure_bytes_available(&f, header_start, header_len, "header")?;
    let mut header = zeroed_payload::<u8>(header_len, "read_npy header")?;
    f.read_exact(&mut header)
        .map_err(|e| format!("header: {e}"))?;
    if !header.ends_with(b"\n") {
        return Err("read_npy: header must end with a newline".into());
    }
    if !header.is_ascii() {
        return Err("read_npy: v1/v2 header must be ASCII".into());
    }
    let header_str = std::str::from_utf8(&header)
        .map_err(|e| format!("read_npy: v1/v2 header is not ASCII: {e}"))?;
    let descr = parse_descr(header_str)?;
    if parse_fortran_order(header_str)? {
        return Err("fortran-order npy not supported".into());
    }
    let shape = parse_shape(header_str)?;
    let numel = checked_shape_numel(&shape, "read_npy")?;
    let payload_start = f
        .stream_position()
        .map_err(|e| format!("read_npy: payload position: {e}"))?;
    match descr.as_str() {
        "<f4" | "|f4" => {
            let payload_bytes =
                checked_payload_nbytes(numel, std::mem::size_of::<f32>(), "read_npy")?;
            ensure_bytes_available(&f, payload_start, payload_bytes, "payload")?;
            let mut data = zeroed_payload::<f32>(numel, "read_npy")?;
            #[cfg(target_endian = "little")]
            {
                // SAFETY: reinterprets an owned, freshly allocated `Vec`'s
                // storage as the byte slice `read_exact` fills. The pointer is
                // valid and uniquely owned for the whole block, the length is
                // `size_of_val` of that same allocation so it cannot overrun,
                // and every bit pattern is a valid `f32`/`i64` — the file may
                // hold nonsense numbers but never an invalid value. Gated on
                // little-endian, where the on-disk layout matches memory.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        data.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of_val(data.as_slice()),
                    )
                };
                f.read_exact(bytes)
                    .map_err(|e| format!("f32 payload: {e}"))?;
            }
            #[cfg(target_endian = "big")]
            for value in &mut data {
                let mut bytes = [0u8; 4];
                f.read_exact(&mut bytes)
                    .map_err(|e| format!("f32 payload: {e}"))?;
                *value = f32::from_le_bytes(bytes);
            }
            Ok(NpyArray {
                shape,
                data_f32: Some(data),
                data_i64: None,
            })
        }
        "<i8" | "|i8" => {
            let payload_bytes =
                checked_payload_nbytes(numel, std::mem::size_of::<i64>(), "read_npy")?;
            ensure_bytes_available(&f, payload_start, payload_bytes, "payload")?;
            let mut data = zeroed_payload::<i64>(numel, "read_npy")?;
            #[cfg(target_endian = "little")]
            {
                // SAFETY: reinterprets an owned, freshly allocated `Vec`'s
                // storage as the byte slice `read_exact` fills. The pointer is
                // valid and uniquely owned for the whole block, the length is
                // `size_of_val` of that same allocation so it cannot overrun,
                // and every bit pattern is a valid `f32`/`i64` — the file may
                // hold nonsense numbers but never an invalid value. Gated on
                // little-endian, where the on-disk layout matches memory.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        data.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of_val(data.as_slice()),
                    )
                };
                f.read_exact(bytes)
                    .map_err(|e| format!("i64 payload: {e}"))?;
            }
            #[cfg(target_endian = "big")]
            for value in &mut data {
                let mut bytes = [0u8; 8];
                f.read_exact(&mut bytes)
                    .map_err(|e| format!("i64 payload: {e}"))?;
                *value = i64::from_le_bytes(bytes);
            }
            Ok(NpyArray {
                shape,
                data_f32: None,
                data_i64: Some(data),
            })
        }
        other => Err(format!("unsupported dtype {other} in {}", path.display())),
    }
}

fn parse_descr(header: &str) -> Result<String, String> {
    // 'descr': '<f4'
    let key = "'descr':";
    let i = header
        .find(key)
        .ok_or_else(|| "missing descr".to_string())?;
    let rest = &header[i + key.len()..];
    let start = rest.find('\'').ok_or_else(|| "descr quote".to_string())? + 1;
    let end = rest[start..]
        .find('\'')
        .ok_or_else(|| "descr end".to_string())?
        + start;
    Ok(rest[start..end].to_string())
}

fn parse_fortran_order(header: &str) -> Result<bool, String> {
    let key = "'fortran_order':";
    let i = header
        .find(key)
        .ok_or_else(|| "missing fortran_order".to_string())?;
    let value = header[i + key.len()..].trim_start();
    let end = value
        .find([',', '}'])
        .ok_or_else(|| "unterminated fortran_order value".to_string())?;
    match value[..end].trim() {
        "True" => Ok(true),
        "False" => Ok(false),
        _ => Err("invalid fortran_order value".into()),
    }
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let key = "'shape':";
    let i = header
        .find(key)
        .ok_or_else(|| "missing shape".to_string())?;
    let rest = &header[i + key.len()..];
    let start = rest.find('(').ok_or_else(|| "shape (".to_string())?;
    let end = rest[start..]
        .find(')')
        .ok_or_else(|| "shape )".to_string())?
        + start;
    let inner = rest[start + 1..end].trim();
    if inner.is_empty() {
        return Ok(vec![]); // scalar
    }
    let mut shape = Vec::new();
    let parts: Vec<_> = inner.split(',').collect();
    for (index, part) in parts.iter().enumerate() {
        let p = part.trim();
        if p.is_empty() {
            if index + 1 == parts.len() && parts.len() > 1 {
                continue; // Python's required singleton-tuple trailing comma.
            }
            return Err("shape contains an empty dimension".into());
        }
        shape.push(
            p.parse::<usize>()
                .map_err(|e| format!("shape parse {p}: {e}"))?,
        );
    }
    Ok(shape)
}

/// Transpose last two dims of a row-major array. `shape` is updated in place.
pub fn transpose_last2(data: &mut [f32], shape: &mut [usize]) -> Result<(), String> {
    if shape.len() < 2 {
        return Err("transpose_last2 needs rank >= 2".into());
    }
    let r = shape.len();
    let rows = shape[r - 2];
    let cols = shape[r - 1];
    let numel = checked_shape_numel(shape, "transpose_last2")?;
    if data.len() != numel {
        return Err(format!(
            "transpose_last2: shape expects {numel} elements, got {}",
            data.len()
        ));
    }
    if numel == 0 {
        shape.swap(r - 2, r - 1);
        return Ok(());
    }
    let matrix = rows
        .checked_mul(cols)
        .ok_or_else(|| "transpose_last2: matrix element count overflow".to_string())?;
    let batch = numel / matrix;
    let mut tmp = zeroed_payload::<f32>(data.len(), "transpose_last2")?;
    for b in 0..batch {
        let start = b * matrix;
        let end = start + matrix;
        let src = &data[start..end];
        let dst = &mut tmp[start..end];
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }
    data.copy_from_slice(&tmp);
    shape.swap(r - 2, r - 1);
    Ok(())
}

/// Write a C-order float32 `.npy` (v1.0).
pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    let numel = checked_shape_numel(shape, "write_npy")?;
    checked_payload_nbytes(numel, std::mem::size_of::<f32>(), "write_npy")?;
    if data.len() != numel {
        return Err(format!(
            "write_npy shape {:?} expects {} elems, got {}",
            shape,
            numel,
            data.len()
        ));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
    // v1.0: magic(6) + ver(2) + hlen(2) + header + '\\n'; header padded so
    // (10 + header.len()) % 64 == 0.
    let mut total = 10 + header.len() + 1;
    let pad = (64 - (total % 64)) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    total = 10 + header.len();
    debug_assert_eq!(total % 64, 0);
    let hlen = u16::try_from(header.len())
        .map_err(|_| "write_npy: v1 header length exceeds u16".to_string())?;

    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(b"\x93NUMPY")
        .map_err(|e| format!("magic: {e}"))?;
    f.write_all(&[1u8, 0]).map_err(|e| format!("ver: {e}"))?;
    f.write_all(&hlen.to_le_bytes())
        .map_err(|e| format!("hlen: {e}"))?;
    f.write_all(header.as_bytes())
        .map_err(|e| format!("header: {e}"))?;
    // macOS/Apple Silicon is little-endian. Writing one four-byte value per
    // syscall made exact-scale checkpoints take minutes per tensor; expose the
    // already-contiguous slice as bytes and submit one bulk payload instead.
    #[cfg(target_endian = "little")]
    {
        // SAFETY: read-only view of a caller-owned `&[f32]` as bytes. The
        // borrow keeps it alive and immutable for the call, the length is
        // `size_of_val` of that same slice, and `u8` has no alignment or
        // validity requirement any `f32` allocation could fail. Gated on
        // little-endian, matching the `<f4` descriptor written above.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        f.write_all(bytes).map_err(|e| format!("payload: {e}"))?;
    }
    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(data));
        for &value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        f.write_all(&bytes).map_err(|e| format!("payload: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempNpy(PathBuf);

    impl TempNpy {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            Self(
                std::env::temp_dir()
                    .join(format!("tessl-npy-{label}-{}-{id}.npy", std::process::id())),
            )
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempNpy {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn write_header_only(path: &Path, shape: &str) {
        let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
        let pad = (64 - ((10 + header.len() + 1) % 64)) % 64;
        header.push_str(&" ".repeat(pad));
        header.push('\n');
        let hlen = u16::try_from(header.len()).expect("small test header");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&hlen.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(path, bytes).expect("write adversarial npy header");
    }

    #[test]
    fn read_rejects_unknown_versions_and_hostile_header_lengths_before_allocation() {
        let unknown = TempNpy::new("unknown-version");
        std::fs::write(unknown.path(), b"\x93NUMPY\x03\x00")
            .expect("write unknown-version preamble");
        let err = read_npy(unknown.path()).expect_err("v3 is outside this reader's contract");
        assert!(err.contains("unsupported npy version 3.0"), "{err}");

        let oversized = TempNpy::new("oversized-header");
        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend_from_slice(
            &u32::try_from(MAX_HEADER_BYTES + 1)
                .expect("test ceiling fits u32")
                .to_le_bytes(),
        );
        std::fs::write(oversized.path(), bytes).expect("write oversized-header preamble");
        let err = read_npy(oversized.path()).expect_err("oversized header must be bounded");
        assert!(err.contains("header length"), "{err}");
        assert!(err.contains("safety bound"), "{err}");

        let truncated = TempNpy::new("truncated-header");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&64u16.to_le_bytes());
        bytes.extend_from_slice(b"too short");
        std::fs::write(truncated.path(), bytes).expect("write truncated-header preamble");
        let err = read_npy(truncated.path()).expect_err("truncated header must be rejected");
        assert!(err.contains("header needs 64 bytes"), "{err}");
    }

    #[test]
    fn read_rejects_non_ascii_and_malformed_shape_tokens() {
        let non_ascii = TempNpy::new("non-ascii-header");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        let header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (), } \xc3\xa9\n";
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header);
        std::fs::write(non_ascii.path(), bytes).expect("write non-ASCII header");
        let err = read_npy(non_ascii.path()).expect_err("NPY v1/v2 headers are ASCII");
        assert!(err.contains("must be ASCII"), "{err}");

        for (label, shape) in [("leading-empty", "(, 2)"), ("middle-empty", "(2, , 3)")] {
            let file = TempNpy::new(label);
            write_header_only(file.path(), shape);
            let err = read_npy(file.path()).expect_err("empty shape token must be rejected");
            assert!(err.contains("empty dimension"), "{shape}: {err}");
        }
    }

    #[test]
    fn read_rejects_shape_product_overflow_without_panicking() {
        let file = TempNpy::new("read-overflow");
        write_header_only(file.path(), &format!("({}, 2)", usize::MAX));

        let outcome = std::panic::catch_unwind(|| read_npy(file.path()));
        let err = outcome
            .expect("read_npy must return Err, not panic")
            .expect_err("overflowing shape must be rejected");
        assert!(err.contains("shape element count overflow"), "{err}");
    }

    #[test]
    fn write_rejects_shape_product_overflow_without_touching_the_path() {
        let file = TempNpy::new("write-overflow");
        let outcome =
            std::panic::catch_unwind(|| write_npy_f32(file.path(), &[usize::MAX, 2], &[]));
        let err = outcome
            .expect("write_npy_f32 must return Err, not panic")
            .expect_err("overflowing shape must be rejected");
        assert!(err.contains("shape element count overflow"), "{err}");
        assert!(
            !file.path().exists(),
            "shape validation must happen before the destination is created"
        );
    }

    #[test]
    fn transpose_rejects_overflow_and_shape_data_mismatch_without_panicking() {
        let mut empty = Vec::new();
        let mut overflowing = vec![usize::MAX, 2, 1, 1];
        let overflow = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            transpose_last2(&mut empty, &mut overflowing)
        }));
        let err = overflow
            .expect("transpose_last2 must return Err, not panic")
            .expect_err("overflowing shape must be rejected");
        assert!(err.contains("shape element count overflow"), "{err}");

        let mut short = vec![0.0; 3];
        let mut shape = vec![2, 2];
        let mismatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            transpose_last2(&mut short, &mut shape)
        }));
        let err = mismatch
            .expect("transpose_last2 must return Err, not panic")
            .expect_err("shape/data mismatch must be rejected");
        assert!(err.contains("expects 4 elements, got 3"), "{err}");
    }

    #[test]
    fn checked_paths_preserve_roundtrip_and_transpose_behavior() {
        let file = TempNpy::new("roundtrip");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        write_npy_f32(file.path(), &[2, 3], &data).expect("write matrix");
        let loaded = read_npy(file.path()).expect("read matrix");
        assert_eq!(loaded.shape, vec![2, 3]);
        assert_eq!(loaded.f32_slice().unwrap(), data);

        let mut transposed = data;
        let mut shape = vec![2, 3];
        transpose_last2(&mut transposed, &mut shape).expect("transpose matrix");
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
