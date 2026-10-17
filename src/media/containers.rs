//! Declared-length checks for audio and video containers, so a cut-short file
//! is an error rather than a shorter clip.
//!
//! AudioToolbox decodes what a truncated WAV, AIFF, CAF or MP3 still holds and
//! returns it as if it were the whole file (a WAV cut in half came back as
//! 3189 of 6400 samples with `noErr`). These containers declare the size of
//! their sample data, so the declaration is compared with the bytes present
//! before the file is handed to the decoder. MP4/M4A boxes are checked the same
//! way, which also catches a cut `mdat` before AVFoundation is asked to read
//! it. FLAC declares its total sample count in STREAMINFO; [`flac_total_samples`]
//! returns it for the caller to compare with what was decoded. An MP3 declares
//! its audio size only when it carries an Xing/Info tag.
//!
//! The checks read headers through [`Source`], so a large file is not loaded
//! to be checked. They validate sizes only: no checksums, no codec state.

use std::fs::File;
use std::os::unix::fs::FileExt;

/// Random access to the bytes of a file or buffer.
pub(crate) trait Source {
    fn len(&self) -> u64;
    /// Fill `buf` from `offset`; an error when fewer bytes exist.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), String>;
}

impl Source for &[u8] {
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), String> {
        let start = usize::try_from(offset).map_err(|_| "offset overflows".to_string())?;
        let end = start.checked_add(buf.len()).ok_or("offset overflows")?;
        let src = self
            .get(start..end)
            .ok_or_else(|| format!("cut inside bytes {start}..{end}"))?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

/// A file and its length when it was opened.
pub(crate) struct FileSource {
    pub file: File,
    pub len: u64,
}

impl Source for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), String> {
        self.file
            .read_exact_at(buf, offset)
            .map_err(|e| format!("reading {} bytes at {offset}: {e}", buf.len()))
    }
}

fn arr<const N: usize>(src: &dyn Source, at: u64) -> Result<[u8; N], String> {
    let mut b = [0u8; N];
    src.read_at(at, &mut b)?;
    Ok(b)
}

/// `Ok(())` when the container's declared sample data fits in the source (or
/// the format is not one of those checked); the message says what is short.
pub(crate) fn check_audio(src: &dyn Source) -> Result<(), String> {
    let Ok(magic) = arr::<12>(src, 0) else {
        return Ok(()); // too short to be any of these; the decoder will say so
    };
    if &magic[..4] == b"RIFF" && &magic[8..12] == b"WAVE" {
        walk_chunks(src, 12, ChunkStyle::RiffLe, b"data", "WAV")
    } else if &magic[..4] == b"FORM" && (&magic[8..12] == b"AIFF" || &magic[8..12] == b"AIFC") {
        walk_chunks(src, 12, ChunkStyle::IffBe, b"SSND", "AIFF")
    } else if &magic[..4] == b"caff" {
        walk_chunks(src, 8, ChunkStyle::Caf, b"data", "CAF")
    } else if &magic[4..8] == b"ftyp" {
        check_mp4_boxes(src)
    } else if let Some(r) = check_mp3_xing(src) {
        r
    } else {
        Ok(())
    }
}

/// `Ok(())` when a video file's top-level boxes are whole (MP4, M4V and
/// QuickTime `.mov`, which may begin with `moov`, `mdat`, `wide` or `free`
/// rather than `ftyp`); other containers are not checked.
pub(crate) fn check_video(src: &dyn Source) -> Result<(), String> {
    let Ok(head) = arr::<8>(src, 0) else {
        return Ok(());
    };
    if matches!(&head[4..8], b"ftyp" | b"moov" | b"mdat" | b"wide" | b"free" | b"skip") {
        check_mp4_boxes(src)
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ChunkStyle {
    /// 4-byte tag, `u32` little-endian size, chunks padded to even sizes.
    RiffLe,
    /// 4-byte tag, `u32` big-endian size, padded to even sizes.
    IffBe,
    /// 4-byte tag, `i64` big-endian size (-1: to the end), no padding.
    Caf,
}

/// Walk the chunks from `start` to the one named `data_tag`, which must fit.
fn walk_chunks(src: &dyn Source, start: u64, style: ChunkStyle, data_tag: &[u8; 4], what: &str) -> Result<(), String> {
    let (header, len) = match style {
        ChunkStyle::Caf => (12u64, src.len()),
        _ => (8u64, src.len()),
    };
    let mut at = start;
    while at + header <= len {
        let mut h = [0u8; 12];
        src.read_at(at, &mut h[..header as usize])?;
        let tag = &h[..4];
        let (size, unknown): (u64, bool) = match style {
            ChunkStyle::RiffLe => {
                let s = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
                (u64::from(s), s == 0xFFFF_FFFF)
            }
            ChunkStyle::IffBe => (u64::from(u32::from_be_bytes([h[4], h[5], h[6], h[7]])), false),
            ChunkStyle::Caf => {
                let s = i64::from_be_bytes([h[4], h[5], h[6], h[7], h[8], h[9], h[10], h[11]]);
                (s.max(0) as u64, s < 0)
            }
        };
        if tag == data_tag {
            // A size of -1 (CAF) or 0xFFFFFFFF (WAV) means "to the end": streamed files.
            if unknown {
                return Ok(());
            }
            let end = at
                .checked_add(header)
                .and_then(|v| v.checked_add(size))
                .ok_or("chunk size overflows")?;
            return if end > len {
                Err(format!(
                    "{what}: the {} chunk declares {size} bytes but the file ends {} bytes short",
                    String::from_utf8_lossy(tag),
                    end - len
                ))
            } else {
                Ok(())
            };
        }
        if unknown {
            return Ok(());
        }
        let pad = if matches!(style, ChunkStyle::Caf) { 0 } else { size & 1 };
        at = at
            .checked_add(header)
            .and_then(|v| v.checked_add(size))
            .and_then(|v| v.checked_add(pad))
            .ok_or("chunk size overflows")?;
    }
    if at > len {
        return Err(format!("{what}: the file is cut inside a chunk before its sample data"));
    }
    Ok(()) // no data chunk: the decoder reports a file with no audio
}

/// Top-level MP4 boxes must be whole.
fn check_mp4_boxes(src: &dyn Source) -> Result<(), String> {
    let len = src.len();
    let mut at = 0u64;
    let mut boxes = 0;
    while at + 8 <= len {
        let h = arr::<8>(src, at)?;
        let size32 = u32::from_be_bytes([h[0], h[1], h[2], h[3]]);
        let tag = String::from_utf8_lossy(&h[4..8]).into_owned();
        let size = match size32 {
            0 => len - at, // the box runs to the end of the file
            1 => u64::from_be_bytes(arr::<8>(src, at + 8)?),
            n => u64::from(n),
        };
        if size < 8 {
            return Err(format!("MP4: box {tag:?} at byte {at} has size {size}"));
        }
        let end = at.checked_add(size).ok_or("MP4: box size overflows")?;
        if end > len {
            return Err(format!(
                "MP4: box {tag:?} at byte {at} declares {size} bytes but the file ends {} bytes short",
                end - len
            ));
        }
        at = end;
        boxes += 1;
        if boxes > 1 << 20 {
            return Err("MP4: too many top-level boxes".into());
        }
    }
    Ok(())
}

/// The front of `src` that holds an MP3's ID3v2 tag and first frame, or `None`
/// when it does not look like one.
pub(crate) fn mp3_prefix(src: &dyn Source) -> Option<Vec<u8>> {
    let mut head = [0u8; 10];
    src.read_at(0, &mut head).ok()?;
    let want = super::mp3::prefix_len(&head).min(src.len() as usize);
    let mut prefix = vec![0u8; want];
    src.read_at(0, &mut prefix).ok()?;
    Some(prefix)
}

/// For an MP3 with an Xing/Info tag that declares the stream size, the file must
/// hold that many bytes after the ID3v2 tag. `None` for anything else.
fn check_mp3_xing(src: &dyn Source) -> Option<Result<(), String>> {
    let prefix = mp3_prefix(src)?;
    let (first_frame, declared) = super::mp3::declared_stream(&prefix)?;
    let need = first_frame.checked_add(declared)?;
    Some(if need > src.len() {
        Err(format!(
            "MP3: the tag declares {declared} bytes of audio from byte {first_frame} but the file ends {} bytes short",
            need - src.len()
        ))
    } else {
        Ok(())
    })
}

/// The total sample count FLAC's STREAMINFO declares; `None` when the source is
/// not FLAC or the count is unknown (0).
pub(crate) fn flac_total_samples(src: &dyn Source) -> Option<u64> {
    let h = arr::<42>(src, 0).ok()?;
    if &h[..4] != b"fLaC" || h[4] & 0x7f != 0 {
        return None; // the first metadata block must be STREAMINFO (type 0)
    }
    // STREAMINFO body from byte 8: 16+16 block sizes, 24+24 frame sizes, then
    // 20 bits rate, 3 channels, 5 bits per sample, 36 bits total samples.
    let total = u64::from_be_bytes([h[18], h[19], h[20], h[21], h[22], h[23], h[24], h[25]]) & ((1 << 36) - 1);
    (total > 0).then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(data_len: u32, present: usize) -> Vec<u8> {
        let mut v = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&[1, 0, 1, 0, 0x80, 0x3e, 0, 0, 0, 0x7d, 0, 0, 2, 0, 16, 0]);
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        v.extend(std::iter::repeat_n(0u8, present));
        v
    }

    #[test]
    fn a_wav_must_hold_the_samples_it_declares() {
        let whole = wav(100, 100);
        assert_eq!(check_audio(&&whole[..]), Ok(()));
        let cut = wav(100, 60);
        assert!(check_audio(&&cut[..]).is_err());
        // Streaming writers declare 0xFFFFFFFF: to the end of the file.
        assert_eq!(check_audio(&&wav(u32::MAX, 10)[..]), Ok(()));
        // Cut before the data chunk starts.
        let mut early = wav(100, 100);
        early.truncate(30);
        assert!(check_audio(&&early[..]).is_err());
    }

    #[test]
    fn mp4_boxes_must_be_whole() {
        let mut v = Vec::new();
        v.extend_from_slice(&16u32.to_be_bytes());
        v.extend_from_slice(b"ftypisom\0\0\0\0");
        v.extend_from_slice(&24u32.to_be_bytes());
        v.extend_from_slice(b"mdat");
        v.extend_from_slice(&[0; 16]);
        assert_eq!(check_audio(&&v[..]), Ok(()));
        assert!(check_audio(&&v[..v.len() - 1]).is_err());
    }

    #[test]
    fn flac_total_samples_are_read_from_streaminfo() {
        let mut v = b"fLaC".to_vec();
        v.extend_from_slice(&[0x00, 0, 0, 34]); // STREAMINFO, 34 bytes
        v.extend_from_slice(&[0x10, 0, 0x10, 0, 0, 0, 0, 0, 0, 0]); // block and frame sizes
                                                                    // 20-bit rate 16000, 3-bit (channels - 1) = 0, 5-bit (bps - 1) = 15, 36-bit total 6400.
        let rate: u64 = 16000;
        let packed: u64 = (rate << 44) | (15 << 36) | 6400;
        v.extend_from_slice(&packed.to_be_bytes());
        v.extend_from_slice(&[0; 16]);
        assert_eq!(flac_total_samples(&&v[..]), Some(6400));
        assert_eq!(flac_total_samples(&&b"RIFF"[..]), None);
    }
}
