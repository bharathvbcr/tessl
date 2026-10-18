//! Structural completeness checks for image containers ImageIO decodes
//! without complaint when they are cut short.
//!
//! ImageIO reports a truncated PNG (all but a trailing CRC), JPEG, GIF or HEIC
//! through `CGImageSourceGetStatusAtIndex`, but a TIFF or an uncompressed BMP
//! with missing pixel data decodes "successfully" with status 0 and zeros where
//! the data should be (measured: 94 of 97 cuts of a TIFF, 96 of 97 of a BMP
//! came back as a picture), and a PNG cut inside the last chunk's CRC does too.
//! A damaged file must be an error, so these formats are checked against what
//! their own headers declare before they are handed to ImageIO.
//!
//! The checks look at sizes and offsets only: they do not decode, validate
//! checksums or interpret colour. A format with no check here is left to
//! ImageIO's status.

/// `Ok(())` when the container's declared extents fit in `bytes`; the message
/// names the first one that does not.
pub(crate) fn check_container(bytes: &[u8]) -> Result<(), String> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        check_png(bytes)
    } else if bytes.starts_with(b"BM") {
        check_bmp(bytes)
    } else if bytes.starts_with(b"II*\0")
        || bytes.starts_with(b"MM\0*")
        || bytes.starts_with(b"II+\0")
        || bytes.starts_with(b"MM\0+")
    {
        check_tiff(bytes)
    } else {
        Ok(())
    }
}

/// Every chunk up to `IEND` (or up to the end of the data, at a chunk
/// boundary) must be complete, CRC included, and image data must have started.
fn check_png(b: &[u8]) -> Result<(), String> {
    let mut at = 8usize;
    let mut saw_idat = false;
    while at < b.len() {
        if b.len() - at < 12 {
            return Err(format!("PNG: cut inside a chunk header at byte {at}"));
        }
        let len = u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;
        let tag = &b[at + 4..at + 8];
        let end = at
            .checked_add(12)
            .and_then(|v| v.checked_add(len))
            .ok_or_else(|| "PNG: chunk length overflows".to_string())?;
        if end > b.len() {
            return Err(format!(
                "PNG: chunk {:?} at byte {at} declares {len} bytes but the file ends {} bytes short",
                String::from_utf8_lossy(tag),
                end - b.len()
            ));
        }
        saw_idat |= tag == b"IDAT";
        if tag == b"IEND" {
            return Ok(());
        }
        at = end;
    }
    if saw_idat {
        Ok(()) // no IEND, but every chunk is whole (PIL reads these too)
    } else {
        Err("PNG: no image data chunk".to_string())
    }
}

/// An uncompressed BMP's pixel array must fit: `bfOffBits + stride * |height|`.
fn check_bmp(b: &[u8]) -> Result<(), String> {
    let u16le = |at: usize| b.get(at..at + 2).map(|s| u64::from(u16::from_le_bytes([s[0], s[1]])));
    let u32le = |at: usize| {
        b.get(at..at + 4)
            .map(|s| u64::from(u32::from_le_bytes([s[0], s[1], s[2], s[3]])))
    };
    let i32le = |at: usize| {
        b.get(at..at + 4)
            .map(|s| i64::from(i32::from_le_bytes([s[0], s[1], s[2], s[3]])))
    };
    let short = || "BMP: the header is cut short".to_string();
    let offset = u32le(10).ok_or_else(short)?;
    let dib = u32le(14).ok_or_else(short)?;
    let (width, height, bpp, compression) = if dib == 12 {
        (
            u16le(18).ok_or_else(short)? as i64,
            u16le(20).ok_or_else(short)? as i64,
            u16le(24).ok_or_else(short)?,
            0u64,
        )
    } else if dib >= 40 {
        (
            i32le(18).ok_or_else(short)?,
            i32le(22).ok_or_else(short)?,
            u16le(28).ok_or_else(short)?,
            u32le(30).ok_or_else(short)?,
        )
    } else {
        return Err(format!("BMP: unknown header size {dib}"));
    };
    // BI_RGB, BI_BITFIELDS, BI_ALPHABITFIELDS are stored row by row; the run
    // length and embedded-image compressions have no size to check.
    if !matches!(compression, 0 | 3 | 6) {
        return Ok(());
    }
    let rows = height.unsigned_abs();
    let stride = (width.unsigned_abs() * bpp).div_ceil(32) * 4;
    let need = rows
        .checked_mul(stride)
        .and_then(|v| v.checked_add(offset))
        .ok_or_else(|| "BMP: size overflows".to_string())?;
    if need > b.len() as u64 {
        return Err(format!(
            "BMP: the pixel array needs {need} bytes ({rows} rows of {stride}) but the file has {}",
            b.len()
        ));
    }
    Ok(())
}

/// The first IFD's strips (or tiles) must lie inside the file.
fn check_tiff(b: &[u8]) -> Result<(), String> {
    let big = b[2] == 43 || b[3] == 43;
    let le = b[0] == b'I';
    let rd = |at: usize, n: usize| -> Result<u64, String> {
        let s = b
            .get(at..at.checked_add(n).ok_or("TIFF: offset overflows")?)
            .ok_or_else(|| format!("TIFF: cut inside the directory at byte {at}"))?;
        let mut v = 0u64;
        for i in 0..n {
            let byte = if le { s[n - 1 - i] } else { s[i] };
            v = (v << 8) | u64::from(byte);
        }
        Ok(v)
    };
    let (ifd, count_size, entry_size, off_size) = if big {
        (rd(8, 8)?, 8usize, 20usize, 8usize)
    } else {
        (rd(4, 4)?, 2usize, 12usize, 4usize)
    };
    let ifd = usize::try_from(ifd).map_err(|_| "TIFF: directory offset overflows".to_string())?;
    let count = usize::try_from(rd(ifd, count_size)?).map_err(|_| "TIFF: entry count overflows".to_string())?;
    if count > 4096 {
        return Err(format!("TIFF: {count} directory entries"));
    }
    let mut offsets: Option<Vec<u64>> = None;
    let mut counts: Option<Vec<u64>> = None;
    for i in 0..count {
        let e = ifd + count_size + i * entry_size;
        let tag = rd(e, 2)?;
        let ty = rd(e + 2, 2)?;
        let n = usize::try_from(rd(e + 4, off_size)?).map_err(|_| "TIFF: value count overflows".to_string())?;
        if !matches!(tag, 273 | 279 | 324 | 325) {
            continue;
        }
        let item = match ty {
            3 => 2usize,  // SHORT
            4 => 4usize,  // LONG
            16 => 8usize, // LONG8 (BigTIFF)
            _ => return Err(format!("TIFF: tag {tag} has type {ty}")),
        };
        if n == 0 || n > 1 << 20 {
            return Err(format!("TIFF: tag {tag} has {n} values"));
        }
        let total = n.checked_mul(item).ok_or("TIFF: value size overflows")?;
        let data_at = if total <= off_size {
            e + 4 + off_size
        } else {
            usize::try_from(rd(e + 4 + off_size, off_size)?).map_err(|_| "TIFF: value offset overflows".to_string())?
        };
        let mut values = Vec::with_capacity(n);
        for k in 0..n {
            values.push(rd(data_at + k * item, item)?);
        }
        match tag {
            273 | 324 => offsets = Some(values),
            _ => counts = Some(values),
        }
    }
    // Single-strip files may omit the byte counts; their extent cannot be
    // checked from the directory.
    if let (Some(offsets), Some(counts)) = (offsets, counts) {
        if offsets.len() != counts.len() {
            return Err(format!(
                "TIFF: {} strip offsets but {} byte counts",
                offsets.len(),
                counts.len()
            ));
        }
        for (i, (o, c)) in offsets.iter().zip(&counts).enumerate() {
            let end = o.checked_add(*c).ok_or("TIFF: strip extent overflows")?;
            if end > b.len() as u64 {
                return Err(format!(
                    "TIFF: strip {i} ends at byte {end} but the file has {} bytes",
                    b.len()
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x2 24-bit BMP: 14 + 40 header bytes and two rows of 8.
    fn bmp() -> Vec<u8> {
        let mut v = vec![b'B', b'M'];
        v.extend_from_slice(&70u32.to_le_bytes());
        v.extend_from_slice(&[0; 4]);
        v.extend_from_slice(&54u32.to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&2i32.to_le_bytes());
        v.extend_from_slice(&2i32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&24u16.to_le_bytes());
        v.extend_from_slice(&[0; 24]); // compression 0, sizes, resolution, colours
        v.extend_from_slice(&[7; 16]);
        v
    }

    #[test]
    fn a_whole_bmp_passes_and_any_cut_of_its_pixels_fails() {
        let b = bmp();
        assert_eq!(check_container(&b), Ok(()));
        for cut in 54..b.len() {
            assert!(check_container(&b[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn png_chunks_must_be_whole() {
        let mut p = b"\x89PNG\r\n\x1a\n".to_vec();
        for (tag, len) in [(b"IHDR", 13usize), (b"IDAT", 5), (b"IEND", 0)] {
            p.extend_from_slice(&(len as u32).to_be_bytes());
            p.extend_from_slice(tag);
            p.extend(std::iter::repeat_n(0u8, len + 4));
        }
        assert_eq!(check_container(&p), Ok(()));
        assert_eq!(
            check_container(&p[..p.len() - 12]),
            Ok(()),
            "a missing IEND at a chunk boundary is accepted"
        );
        assert!(check_container(&p[..p.len() - 14]).is_err(), "cut inside the IDAT CRC");
        assert!(check_container(&p[..p.len() - 20]).is_err());
        assert!(check_container(&p[..8 + 25]).is_err(), "no image data");
    }

    #[test]
    fn unknown_formats_are_left_to_imageio() {
        assert_eq!(check_container(b"\xff\xd8\xff\xe0 anything"), Ok(()));
        assert_eq!(check_container(&[]), Ok(()));
    }
}
