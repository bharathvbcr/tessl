//! Native media decoding for EmbeddingGemma 2: an image, audio or video file
//! (or its bytes) to the raw inputs [`crate::embedgemma2_pre`] takes, with no
//! Python and no extra crates. macOS only, through the system frameworks:
//!
//! | Media | Framework | Output |
//! |---|---|---|
//! | [`image`] | ImageIO + CoreGraphics | 8-bit RGB or RGBA, HWC ([`image::DecodedImage`]) |
//! | [`audio`] | AudioToolbox `ExtAudioFile`, then [`resample`] | mono `f32` at 16 kHz ([`audio::DecodedAudio`]) |
//! | [`video`] | AVFoundation `AVAssetReader` | sampled RGB frames, fps, timestamps ([`video::DecodedVideo`]) |
//!
//! ```no_run
//! use std::path::Path;
//! use tessl::embedgemma2_pre::{preprocess_audio, preprocess_image, PreConfig};
//! use tessl::media::{audio, image};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = PreConfig::from_snapshot(Path::new("/path/to/snapshot"))?;
//! let img = image::decode_image_file(Path::new("cat.png"), &image::ImageOptions::default())?;
//! let prepared = preprocess_image(&cfg.image, &img.raw())?;
//! let clip = audio::decode_audio_file(Path::new("speech.m4a"), &audio::AudioOptions::default())?;
//! let features = preprocess_audio(&cfg.audio, &clip.samples, clip.sample_rate)?;
//! # let _ = (prepared, features);
//! # Ok(())
//! # }
//! ```
//!
//! Every function returns a [`MediaError`]; a file that cannot be decoded is an
//! error, never an empty result. Every Core Foundation object is held by an RAII
//! guard ([`cf::CfRef`]) and released on every path. The declarations of the C
//! functions are this crate's own, written from the macOS SDK headers; no
//! Objective-C binding crate is used (AVFoundation goes through `objc2`'s
//! `msg_send!`, which the crate already depends on).

use std::fmt;
use std::path::{Path, PathBuf};

pub mod audio;
pub mod cf;
pub mod codec_config;
mod containers;
pub mod image;
mod integrity;
mod mp3;
mod orient;
pub mod resample;
pub mod video;
mod vt;
pub mod yuv;

pub use audio::{decode_audio_bytes, decode_audio_file, AudioContainer, AudioOptions, DecodedAudio, Downmix};
pub use image::{decode_image, decode_image_file, DecodedImage, ImageOptions};
pub use video::{decode_video_file, preprocess_decoded_video, DecodedVideo, VideoOptions, VideoSource};

/// Why a file or buffer could not be turned into model input.
#[derive(Debug)]
pub enum MediaError {
    /// The file could not be read.
    Io { path: PathBuf, source: std::io::Error },
    /// There were no bytes.
    Empty { what: &'static str },
    /// The system does not recognize the bytes as this kind of media.
    Unrecognized { what: &'static str, detail: String },
    /// The system recognizes the container but not what is inside it.
    Unsupported { what: &'static str, detail: String },
    /// The data is damaged or truncated.
    Decode { what: &'static str, detail: String },
    /// An input is larger than a bound allows.
    Limit { what: &'static str, detail: String },
    /// An argument is unusable.
    Invalid { what: &'static str, detail: String },
    /// A framework call failed in a way none of the above names.
    Framework { api: &'static str, detail: String },
    /// The decoded media was refused by preprocessing.
    Preprocess(String),
}

impl MediaError {
    pub(crate) fn unrecognized(what: &'static str, detail: impl Into<String>) -> Self {
        MediaError::Unrecognized {
            what,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaError::Io { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            MediaError::Empty { what } => write!(f, "{what}: no bytes to decode"),
            MediaError::Unrecognized { what, detail } => {
                write!(f, "{what}: not a format the system decodes ({detail})")
            }
            MediaError::Unsupported { what, detail } => write!(f, "{what}: unsupported ({detail})"),
            MediaError::Decode { what, detail } => write!(f, "{what}: cannot be decoded ({detail})"),
            MediaError::Limit { what, detail } => write!(f, "{what}: over a limit ({detail})"),
            MediaError::Invalid { what, detail } => write!(f, "{what}: invalid argument ({detail})"),
            MediaError::Framework { api, detail } => write!(f, "{api} failed: {detail}"),
            MediaError::Preprocess(msg) => write!(f, "preprocessing refused the decoded media: {msg}"),
        }
    }
}

impl std::error::Error for MediaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MediaError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<MediaError> for String {
    fn from(e: MediaError) -> String {
        e.to_string()
    }
}

/// Read a whole file, refusing an empty one and one over `limit` bytes.
pub(crate) fn read_file_bounded(path: &Path, limit: u64, what: &'static str) -> Result<Vec<u8>, MediaError> {
    let io = |source| MediaError::Io {
        path: path.to_path_buf(),
        source,
    };
    let meta = std::fs::metadata(path).map_err(io)?;
    if !meta.is_file() {
        return Err(MediaError::Invalid {
            what,
            detail: format!("{} is not a file", path.display()),
        });
    }
    if meta.len() == 0 {
        return Err(MediaError::Empty { what });
    }
    if meta.len() > limit {
        return Err(MediaError::Limit {
            what,
            detail: format!("{} bytes exceeds {limit}", meta.len()),
        });
    }
    // Read through a bounded reader so a file that grows after the check cannot
    // exceed the limit.
    use std::io::Read;
    let mut out = Vec::with_capacity(meta.len() as usize);
    std::fs::File::open(path)
        .map_err(io)?
        .take(limit + 1)
        .read_to_end(&mut out)
        .map_err(io)?;
    if out.len() as u64 > limit {
        return Err(MediaError::Limit {
            what,
            detail: format!("more than {limit} bytes"),
        });
    }
    if out.is_empty() {
        return Err(MediaError::Empty { what });
    }
    Ok(out)
}

/// An `OSStatus` as the four-character code it spells, if it spells one, else
/// the number.
fn describe_status(status: i32) -> String {
    let bytes = (status as u32).to_be_bytes();
    if bytes.iter().all(|b| (0x20..0x7f).contains(b)) {
        format!("'{}' ({status})", String::from_utf8_lossy(&bytes))
    } else {
        status.to_string()
    }
}

/// The [`MediaError`] for a failed AudioToolbox call. The codes are the
/// `kAudioFile*Error` constants of `AudioFile.h` and the
/// `kExtAudioFileError_*` ones of `ExtendedAudioFile.h`.
pub(crate) fn os_status_error(what: &'static str, api: &'static str, status: i32) -> MediaError {
    let code = describe_status(status);
    let detail = format!("{api}: OSStatus {code}");
    match &(status as u32).to_be_bytes() {
        // kAudioFileUnsupportedFileTypeError
        b"typ?" => MediaError::Unrecognized { what, detail },
        // kAudioFileUnsupportedDataFormatError
        b"fmt?" => MediaError::Unsupported { what, detail },
        // kAudioFileInvalidFileError, InvalidChunkError, InvalidPacketOffsetError
        b"dta?" | b"chk?" | b"pck?" => MediaError::Decode { what, detail },
        _ => match status {
            // kAudioFileEndOfFileError, kAudioFilePositionError
            -39 | -40 => MediaError::Decode { what, detail },
            _ => MediaError::Framework { api, detail: code },
        },
    }
}
