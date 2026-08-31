//! Minimal NumPy `.npy` reader (v1.0 / v2.0, C-order, f32 / i64 / f64 scalar).

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

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
    let header_len: usize = if ver[0] == 1 {
        let mut hl = [0u8; 2];
        f.read_exact(&mut hl).map_err(|e| format!("hlen: {e}"))?;
        u16::from_le_bytes(hl) as usize
    } else {
        let mut hl = [0u8; 4];
        f.read_exact(&mut hl).map_err(|e| format!("hlen: {e}"))?;
        u32::from_le_bytes(hl) as usize
    };
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .map_err(|e| format!("header: {e}"))?;
    let header_str = String::from_utf8_lossy(&header);
    let descr = parse_descr(&header_str)?;
    let fortran = header_str.contains("fortran_order': True");
    if fortran {
        return Err("fortran-order npy not supported".into());
    }
    let shape = parse_shape(&header_str)?;
    let numel: usize = shape.iter().product();
    match descr.as_str() {
        "<f4" | "|f4" => {
            let mut data = vec![0.0f32; numel];
            #[cfg(target_endian = "little")]
            {
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
            let mut data = vec![0i64; numel];
            #[cfg(target_endian = "little")]
            {
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
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
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
    let numel: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
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

    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(b"\x93NUMPY")
        .map_err(|e| format!("magic: {e}"))?;
    f.write_all(&[1u8, 0]).map_err(|e| format!("ver: {e}"))?;
    let hlen = header.len() as u16;
    f.write_all(&hlen.to_le_bytes())
        .map_err(|e| format!("hlen: {e}"))?;
    f.write_all(header.as_bytes())
        .map_err(|e| format!("header: {e}"))?;
    // macOS/Apple Silicon is little-endian. Writing one four-byte value per
    // syscall made exact-scale checkpoints take minutes per tensor; expose the
    // already-contiguous slice as bytes and submit one bulk payload instead.
    #[cfg(target_endian = "little")]
    {
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

#[allow(dead_code)]
pub fn seek_noop() {}
