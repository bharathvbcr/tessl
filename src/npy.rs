//! Minimal NumPy `.npy` reader and writer (read v1.0 / v2.0 / v3.0, write
//! v1.0; C-order, `<f4`, `<f8` and `<i8`).
//!
//! This exists for benchmark parity, not as a general-purpose format library.
//! The cross-runtime benchmarks compare this crate against PyTorch and MLX on
//! the same operands, and "the same operands" has to mean the same bytes: a
//! lane that generated its own random matrix would be comparing two different
//! problems and calling the difference a speedup. `bench_gemm_sweep` dumps its
//! operands here and the Python lanes read them back.
//!
//! Scope is deliberately narrow. C-order only, no structured dtypes, no
//! Fortran order, no object arrays. An unsupported header is an error rather
//! than a best-effort parse, because a benchmark that silently transposed its
//! input would produce a plausible number for the wrong problem.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::plain::{self, PlainScalar};

#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data_f32: Option<Vec<f32>>,
    pub data_i64: Option<Vec<i64>>,
    pub data_f64: Option<Vec<f64>>,
}

/// Fills `dst` from the reader with the little-endian payload as it sits on
/// disk, straight into `dst`'s storage.
fn read_le_payload<T: PlainScalar>(f: &mut File, dst: &mut [T], what: &str) -> Result<(), String> {
    f.read_exact(plain::bytes_mut(dst))
        .map_err(|e| format!("{what} payload: {e}"))?;
    plain::le_to_native(dst);
    Ok(())
}

impl NpyArray {
    pub fn f32_slice(&self) -> Result<&[f32], String> {
        self.data_f32.as_deref().ok_or_else(|| "expected float32 npy".into())
    }

    /// The f64 payload. Separate from [`NpyArray::f32_slice`] on purpose: a
    /// published GDN golden is generated in f64 precisely so the reference is
    /// not itself a source of error, and silently narrowing it to f32 on load
    /// would discard the property it exists to provide.
    pub fn f64_slice(&self) -> Result<&[f64], String> {
        self.data_f64.as_deref().ok_or_else(|| "expected float64 npy".into())
    }

    pub fn i64_slice(&self) -> Result<&[i64], String> {
        self.data_i64.as_deref().ok_or_else(|| "expected int64 npy".into())
    }

    pub fn scalar_f32(&self) -> Result<f32, String> {
        let s = self.f32_slice()?;
        if s.len() != 1 {
            return Err(format!("expected scalar, got {} elems", s.len()));
        }
        Ok(s[0])
    }
}

/// Largest header accepted. NumPy's own reader refuses headers over 10 000
/// bytes by default; this is looser, and still keeps a hostile length field
/// from allocating gigabytes before anything is checked.
pub const MAX_NPY_HEADER_BYTES: usize = 1 << 20;

pub fn read_npy(path: &Path) -> Result<NpyArray, String> {
    let what = path.display();
    let mut f = File::open(path).map_err(|e| format!("open {what}: {e}"))?;
    let file_len = f.metadata().map_err(|e| format!("{what}: {e}"))?.len();
    let mut magic = [0u8; 6];
    f.read_exact(&mut magic).map_err(|e| format!("read magic: {e}"))?;
    if &magic != b"\x93NUMPY" {
        return Err(format!("not an npy file: {what}"));
    }
    let mut ver = [0u8; 2];
    f.read_exact(&mut ver).map_err(|e| format!("ver: {e}"))?;
    // 1.0: u16 header length; 2.0 and 3.0 (UTF-8 header): u32.
    let (header_len, prefix) = match ver[0] {
        1 => {
            let mut hl = [0u8; 2];
            f.read_exact(&mut hl).map_err(|e| format!("hlen: {e}"))?;
            (u16::from_le_bytes(hl) as usize, 10u64)
        }
        2 | 3 => {
            let mut hl = [0u8; 4];
            f.read_exact(&mut hl).map_err(|e| format!("hlen: {e}"))?;
            (u32::from_le_bytes(hl) as usize, 12u64)
        }
        v => return Err(format!("{what}: unsupported npy format version {v}.{}", ver[1])),
    };
    if header_len > MAX_NPY_HEADER_BYTES || header_len as u64 > file_len.saturating_sub(prefix) {
        return Err(format!(
            "{what}: header length {header_len} exceeds the cap {MAX_NPY_HEADER_BYTES} or the file"
        ));
    }
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header).map_err(|e| format!("header: {e}"))?;
    let header_str = std::str::from_utf8(&header).map_err(|e| format!("{what}: header is not UTF-8: {e}"))?;
    let descr = parse_descr(header_str)?;
    if parse_fortran_order(header_str)? {
        return Err("fortran-order npy not supported".into());
    }
    let shape = parse_shape(header_str)?;
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("{what}: shape {shape:?} overflows usize"))?;
    let elem_size: u64 = match descr.as_str() {
        "<f4" | "|f4" => 4,
        "<f8" | "|f8" | "<i8" | "|i8" => 8,
        other => return Err(format!("unsupported dtype {other} in {what}")),
    };
    // The payload must be exactly the rest of the file: checked before the
    // allocation, so a tiny file declaring a huge shape is an error rather than
    // an abort, and a truncated or padded file is not read as a smaller array.
    let payload = (numel as u64)
        .checked_mul(elem_size)
        .ok_or_else(|| format!("{what}: payload size overflows"))?;
    let rest = file_len - prefix - header_len as u64;
    if payload != rest {
        return Err(format!(
            "{what}: shape {shape:?} x {elem_size} bytes is {payload} bytes, but {rest} follow the header"
        ));
    }
    match descr.as_str() {
        "<f4" | "|f4" => {
            let mut data = vec![0.0f32; numel];
            read_le_payload(&mut f, &mut data, "f32")?;
            Ok(NpyArray {
                shape,
                data_f32: Some(data),
                data_i64: None,
                data_f64: None,
            })
        }
        "<f8" | "|f8" => {
            let mut data = vec![0.0f64; numel];
            read_le_payload(&mut f, &mut data, "f64")?;
            Ok(NpyArray {
                shape,
                data_f32: None,
                data_i64: None,
                data_f64: Some(data),
            })
        }
        "<i8" | "|i8" => {
            let mut data = vec![0i64; numel];
            read_le_payload(&mut f, &mut data, "i64")?;
            Ok(NpyArray {
                shape,
                data_f32: None,
                data_i64: Some(data),
                data_f64: None,
            })
        }
        other => Err(format!("unsupported dtype {other} in {}", path.display())),
    }
}

fn parse_descr(header: &str) -> Result<String, String> {
    // 'descr': '<f4'
    let key = "'descr':";
    let i = header.find(key).ok_or_else(|| "missing descr".to_string())?;
    let rest = &header[i + key.len()..];
    let start = rest.find('\'').ok_or_else(|| "descr quote".to_string())? + 1;
    let end = rest[start..].find('\'').ok_or_else(|| "descr end".to_string())? + start;
    Ok(rest[start..end].to_string())
}

/// The `'fortran_order'` value: exactly `True` or `False`, and required.
fn parse_fortran_order(header: &str) -> Result<bool, String> {
    let key = "'fortran_order':";
    let i = header.find(key).ok_or_else(|| "missing fortran_order".to_string())?;
    let rest = header[i + key.len()..].trim_start();
    if rest.starts_with("False") {
        Ok(false)
    } else if rest.starts_with("True") {
        Ok(true)
    } else {
        Err(format!(
            "fortran_order is neither True nor False: {:?}",
            &rest[..rest.len().min(16)]
        ))
    }
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let key = "'shape':";
    let i = header.find(key).ok_or_else(|| "missing shape".to_string())?;
    let rest = &header[i + key.len()..];
    let start = rest.find('(').ok_or_else(|| "shape (".to_string())?;
    let end = rest[start..].find(')').ok_or_else(|| "shape )".to_string())? + start;
    let inner = rest[start + 1..end].trim();
    if inner.is_empty() {
        return Ok(vec![]); // scalar
    }
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        shape.push(p.parse::<usize>().map_err(|e| format!("shape parse {p}: {e}"))?);
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
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("transpose_last2: shape {shape:?} overflows usize"))?;
    if numel != data.len() {
        return Err(format!(
            "transpose_last2: shape {shape:?} holds {numel} elements, data has {}",
            data.len()
        ));
    }
    let batch: usize = shape[..r - 2].iter().product();
    let mut tmp = vec![0.0f32; data.len()];
    for b in 0..batch {
        let src = &data[b * rows * cols..(b + 1) * rows * cols];
        let dst = &mut tmp[b * rows * cols..(b + 1) * rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }
    data.copy_from_slice(&tmp);
    shape[r - 2] = cols;
    shape[r - 1] = rows;
    Ok(())
}

/// Write a C-order float32 `.npy` (v1.0).
pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("write_npy shape {shape:?} overflows usize"))?;
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
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
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

    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(b"\x93NUMPY").map_err(|e| format!("magic: {e}"))?;
    f.write_all(&[1u8, 0]).map_err(|e| format!("ver: {e}"))?;
    let hlen =
        u16::try_from(header.len()).map_err(|_| format!("write_npy: a {}-byte header needs npy v2", header.len()))?;
    f.write_all(&hlen.to_le_bytes()).map_err(|e| format!("hlen: {e}"))?;
    f.write_all(header.as_bytes()).map_err(|e| format!("header: {e}"))?;
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
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
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

#[allow(dead_code)]
pub fn seek_noop() {}
