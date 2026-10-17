//! The sample layout an H.264 or HEVC stream declares, from its codec
//! configuration record (`avcC` / `hvcC`, the atoms of the MP4 sample
//! description).
//!
//! The pixel format a decoder is *asked* for says nothing about the stream: a
//! 10-bit stream asked for an 8-bit format is converted to it by the system
//! without complaint. Which format VideoToolbox must be asked for is therefore
//! read from the stream: the SPS' `chroma_format_idc` and bit depths (H.264:
//! parsed here from the first SPS of the `avcC`) or the `hvcC`'s own fields.

use super::yuv::{Chroma, Layout};

/// MSB-first bit reader over an SPS payload, with the emulation prevention bytes
/// (`00 00 03`) already removed.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> Result<u32, String> {
        let byte = self
            .data
            .get(self.pos / 8)
            .ok_or("the SPS ends inside its profile fields")?;
        let bit = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Ok(u32::from(bit))
    }

    /// An unsigned Exp-Golomb code `ue(v)`.
    fn ue(&mut self) -> Result<u32, String> {
        let mut zeros = 0u32;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err("an Exp-Golomb code in the SPS is longer than 32 bits".into());
            }
        }
        let mut value = 0u32;
        for _ in 0..zeros {
            value = (value << 1) | self.bit()?;
        }
        Ok((1u32 << zeros) - 1 + value)
    }
}

/// `data` without the `00 00 03` emulation prevention bytes.
fn unescape(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0;
    for &b in data {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

fn layout(chroma_format_idc: u32, luma_depth: u32, chroma_depth: u32, codec: &str) -> Result<Layout, String> {
    let chroma = match chroma_format_idc {
        0 => return Err(format!("monochrome (4:0:0) {codec}")),
        1 => Chroma::C420,
        2 => Chroma::C422,
        3 => Chroma::C444,
        other => return Err(format!("{codec} chroma_format_idc {other}")),
    };
    if luma_depth != chroma_depth {
        return Err(format!(
            "{codec} with {luma_depth}-bit luma and {chroma_depth}-bit chroma"
        ));
    }
    match luma_depth {
        8 | 10 => Ok(Layout {
            chroma,
            depth: luma_depth as u8,
        }),
        other => Err(format!("{other}-bit {codec}")),
    }
}

/// The layout an `avcC` record declares (ISO/IEC 14496-15), from the
/// `chroma_format_idc` and bit depths of its first SPS (the High, High 10, High
/// 4:2:2 and High 4:4:4 profiles carry them; the others are 8-bit 4:2:0).
///
/// # Errors
/// Why the stream has no layout the exact conversion handles: no SPS, an SPS cut
/// short, monochrome, separately coded colour planes, or a bit depth other than 8
/// or 10.
pub fn avc_layout(avcc: &[u8]) -> Result<Layout, String> {
    if avcc.len() < 8 || avcc[0] != 1 {
        return Err("the avcC record is malformed".into());
    }
    if avcc[5] & 0x1F == 0 {
        return Err("the avcC record holds no SPS".into());
    }
    let len = usize::from(u16::from_be_bytes([avcc[6], avcc[7]]));
    let sps = avcc.get(8..8 + len).ok_or("the avcC record's first SPS is cut short")?;
    // NAL header, profile_idc, constraint flags, level_idc, then the bit fields.
    if sps.len() < 5 || sps[0] & 0x1F != 7 {
        return Err("the avcC record's first NAL unit is not an SPS".into());
    }
    let profile = sps[1];
    // The profiles whose SPS carries chroma_format_idc and the bit depths.
    if !matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        return Ok(Layout {
            chroma: Chroma::C420,
            depth: 8,
        });
    }
    let body = unescape(&sps[4..]);
    let mut bits = Bits { data: &body, pos: 0 };
    bits.ue()?; // seq_parameter_set_id
    let chroma_format_idc = bits.ue()?;
    if chroma_format_idc == 3 && bits.bit()? == 1 {
        return Err("H.264 with separately coded colour planes".into());
    }
    let luma = bits.ue()? + 8;
    let chroma = bits.ue()? + 8;
    layout(chroma_format_idc, luma, chroma, "H.264")
}

/// The layout an `hvcC` record declares (`chromaFormat`, `bitDepthLumaMinus8`,
/// `bitDepthChromaMinus8`).
///
/// # Errors
/// Why the stream has no layout the exact conversion handles: a malformed
/// record, monochrome, or a bit depth other than 8 or 10.
pub fn hevc_layout(hvcc: &[u8]) -> Result<Layout, String> {
    if hvcc.len() < 19 || hvcc[0] != 1 {
        return Err("the hvcC record is malformed".into());
    }
    layout(
        u32::from(hvcc[16] & 3),
        u32::from(hvcc[17] & 7) + 8,
        u32::from(hvcc[18] & 7) + 8,
        "HEVC",
    )
}
