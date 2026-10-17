//! The gapless-playback and size numbers of an MP3 file, read from its
//! Xing/Info/LAME tag.
//!
//! An MP3 decoder returns the encoder's delay (silence in front of the signal)
//! and the zero padding that fills the last frame along with the audio.
//! LAME and FFmpeg write both counts into the first frame, in the "Xing" or
//! "Info" header's LAME extension. AudioToolbox uses them to trim the decode
//! of MPEG-1 files (44.1 and 48 kHz: `afinfo` shows "576 priming + 1404
//! remainder"), but not of MPEG-2 and 2.5 files (16, 22.05, 24 kHz and below),
//! whose tag sits at a different offset: a measured 16 kHz mono file decoded to
//! 17280 frames for 16000 of signal, with the signal 576 frames late.
//! [`gapless`] reads the tag so the decoder can trim them itself, and
//! [`declared_stream`] reads the stream size the same tag declares, so a cut-short
//! file is recognised.

/// What the first frame's Xing/Info tag says.
struct Tag {
    /// Byte offset of the first MPEG audio frame.
    frame_at: usize,
    /// Declared size of the audio stream in bytes (the Xing frame included).
    stream_bytes: Option<u32>,
    /// `(encoder delay, padding)` from the LAME extension.
    gapless: Option<(u32, u32)>,
}

/// How many bytes from the start of a file `parse` needs: an ID3v2 tag, if one
/// is flagged in `head` (the first ten bytes), plus room for a frame header and
/// the tag after it. Capped at 8 MiB.
pub(crate) fn prefix_len(head: &[u8]) -> usize {
    const SLACK: usize = 4096;
    let id3 = if head.len() >= 10 && &head[..3] == b"ID3" {
        let size = head[6..10]
            .iter()
            .fold(0usize, |acc, &b| (acc << 7) | usize::from(b & 0x7f));
        10 + size + if head[5] & 0x10 != 0 { 10 } else { 0 }
    } else {
        0
    };
    (id3 + SLACK).min(8 << 20)
}

fn parse(bytes: &[u8]) -> Option<Tag> {
    let mut at = 0usize;
    if bytes.len() >= 10 && &bytes[..3] == b"ID3" {
        // Four 7-bit bytes; a footer, if flagged, adds 10 more.
        let size = bytes[6..10]
            .iter()
            .fold(0usize, |acc, &b| (acc << 7) | usize::from(b & 0x7f));
        let footer = if bytes[5] & 0x10 != 0 { 10 } else { 0 };
        at = 10usize.checked_add(size)?.checked_add(footer)?;
    }
    // The first frame header: eleven sync bits.
    let mut found = None;
    while at + 4 <= bytes.len() && at < (8 << 20) {
        if bytes[at] == 0xFF && bytes[at + 1] & 0xE0 == 0xE0 {
            found = Some(at);
            break;
        }
        at += 1;
    }
    let frame_at = found?;
    let b1 = bytes[frame_at + 1];
    let version = (b1 >> 3) & 3; // 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5, 1 reserved
    let layer = (b1 >> 1) & 3; // 1 = Layer III
    if version == 1 || layer != 1 {
        return None;
    }
    let protected = b1 & 1 == 0; // a CRC follows the header
    let mono = (bytes[frame_at + 3] >> 6) & 3 == 3;
    let side_info = match (version, mono) {
        (3, true) => 17,
        (3, false) => 32,
        (_, true) => 9,
        (_, false) => 17,
    };
    let tag = frame_at + 4 + if protected { 2 } else { 0 } + side_info;
    let name = bytes.get(tag..tag + 4)?;
    if name != b"Xing" && name != b"Info" {
        return None;
    }
    let flags = u32::from_be_bytes(bytes.get(tag + 4..tag + 8)?.try_into().ok()?);
    let mut pos = tag + 8;
    if flags & 1 != 0 {
        pos += 4; // frame count
    }
    let mut stream_bytes = None;
    if flags & 2 != 0 {
        stream_bytes = Some(u32::from_be_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?));
        pos += 4;
    }
    if flags & 4 != 0 {
        pos += 100; // table of contents
    }
    if flags & 8 != 0 {
        pos += 4; // quality
    }
    // The LAME extension: 9 bytes of encoder name, then revision, lowpass,
    // replay gain, flags and bitrate fill 12 more, then 12 bits of delay and 12
    // of padding.
    let gapless = bytes.get(pos..pos + 24).and_then(|ext| {
        if !(ext.starts_with(b"LAME") || ext.starts_with(b"Lavc") || ext.starts_with(b"Lavf")) {
            return None;
        }
        let (d, e, f) = (u32::from(ext[21]), u32::from(ext[22]), u32::from(ext[23]));
        Some(((d << 4) | (e >> 4), ((e & 0xF) << 8) | f))
    });
    Some(Tag {
        frame_at,
        stream_bytes,
        gapless,
    })
}

/// `(encoder delay, padding)` in frames, from the LAME extension of the Xing or
/// Info tag in the first MPEG audio frame of `bytes` (an optional ID3v2 tag may
/// come first). `None` when there is no such tag.
pub(crate) fn gapless(bytes: &[u8]) -> Option<(u32, u32)> {
    parse(bytes)?.gapless
}

/// `(offset of the first frame, declared audio bytes)` when the Xing/Info tag
/// declares the stream size.
pub(crate) fn declared_stream(bytes: &[u8]) -> Option<(u64, u64)> {
    let tag = parse(bytes)?;
    Some((tag.frame_at as u64, u64::from(tag.stream_bytes?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An MPEG-2 Layer III mono frame header with an Info tag carrying
    /// `delay` and `padding`, after an ID3v2 tag of `id3` payload bytes.
    fn file(id3: usize, delay: u32, padding: u32) -> Vec<u8> {
        let mut v = Vec::new();
        if id3 > 0 {
            v.extend_from_slice(b"ID3\x04\x00\x00");
            v.extend_from_slice(&[0, 0, (id3 >> 7) as u8 & 0x7f, id3 as u8 & 0x7f]);
            v.extend(std::iter::repeat_n(0u8, id3));
        }
        // 0xFFF3: MPEG-2, Layer III, no CRC; 0x90 bitrate/rate; mono (11 << 6).
        v.extend_from_slice(&[0xFF, 0xF3, 0x90, 0xC4]);
        v.extend(std::iter::repeat_n(0u8, 9)); // side info, mono MPEG-2
        v.extend_from_slice(b"Info");
        v.extend_from_slice(&3u32.to_be_bytes()); // flags: frame count and byte count
        v.extend_from_slice(&30u32.to_be_bytes());
        v.extend_from_slice(&8640u32.to_be_bytes());
        v.extend_from_slice(b"Lavc63.1."); // 9 bytes of encoder name
        v.extend_from_slice(&[0; 12]);
        v.extend_from_slice(&[
            (delay >> 4) as u8,
            ((delay & 0xF) << 4 | (padding >> 8)) as u8,
            padding as u8,
        ]);
        v
    }

    #[test]
    fn the_lame_extension_is_read_after_an_id3_tag_or_without_one() {
        assert_eq!(gapless(&file(0, 576, 704)), Some((576, 704)));
        assert_eq!(gapless(&file(300, 1105, 1404)), Some((1105, 1404)));
        assert_eq!(gapless(&file(0, 0, 0)), Some((0, 0)));
    }

    #[test]
    fn the_declared_stream_size_and_the_prefix_to_read_come_from_the_same_tag() {
        assert_eq!(declared_stream(&file(0, 576, 704)), Some((0, 8640)));
        let with_id3 = file(300, 576, 704);
        assert_eq!(declared_stream(&with_id3), Some((310, 8640)));
        assert_eq!(prefix_len(&with_id3[..10]), 310 + 4096);
        assert_eq!(prefix_len(b"RIFF....WAVE"), 4096);
    }

    #[test]
    fn anything_else_has_no_tag() {
        assert_eq!(gapless(&[]), None);
        assert_eq!(gapless(b"RIFF....WAVEfmt "), None);
        assert_eq!(declared_stream(b"RIFF....WAVEfmt "), None);
        let mut cut = file(0, 576, 704);
        cut.truncate(cut.len() - 5);
        assert_eq!(gapless(&cut), None, "a tag cut short");
        let mut other = file(0, 576, 704);
        other[13] = b'X'; // "Info" -> "Xnfo"
        assert_eq!(gapless(&other), None);
    }
}
