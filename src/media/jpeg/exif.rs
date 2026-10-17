//! The EXIF orientation of an APP1 segment.

fn u16_at(t: &[u8], at: usize, le: bool) -> Option<u32> {
    let b: [u8; 2] = t.get(at..at.checked_add(2)?)?.try_into().ok()?;
    Some(u32::from(if le {
        u16::from_le_bytes(b)
    } else {
        u16::from_be_bytes(b)
    }))
}

fn u32_at(t: &[u8], at: usize, le: bool) -> Option<u32> {
    let b: [u8; 4] = t.get(at..at.checked_add(4)?)?.try_into().ok()?;
    Some(if le {
        u32::from_le_bytes(b)
    } else {
        u32::from_be_bytes(b)
    })
}

/// The `Orientation` (tag 0x0112) of IFD0 in an Exif APP1 payload (`Exif\0\0`
/// then a TIFF header), 1..=8; `None` when there is none or it is malformed.
/// Every offset is bounds-checked against the payload.
pub(crate) fn orientation(app1: &[u8]) -> Option<u32> {
    let t = app1.strip_prefix(b"Exif\0\0")?;
    let le = match t.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    if u16_at(t, 2, le)? != 42 {
        return None;
    }
    let ifd = usize::try_from(u32_at(t, 4, le)?).ok()?;
    let n = u16_at(t, ifd, le)? as usize;
    for i in 0..n {
        let e = ifd.checked_add(2)?.checked_add(i.checked_mul(12)?)?;
        if u16_at(t, e, le)? != 0x0112 {
            continue;
        }
        let (ty, count) = (u16_at(t, e + 2, le)?, u32_at(t, e + 4, le)?);
        if ty != 3 || count != 1 {
            return None;
        }
        let v = u16_at(t, e + 8, le)?;
        return (1..=8).contains(&v).then_some(v);
    }
    None
}
