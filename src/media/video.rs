//! Video decoding with AVFoundation and VideoToolbox.
//!
//! H.264, HEVC and whatever else AVFoundation reads in MP4, MOV and M4V become
//! decoded 8-bit RGB frames, sampled the way
//! [`crate::embedgemma2_pre::sample_frame_indices`] samples them, with the
//! frame rate and duration the reference loader computes and each frame's
//! presentation time. For the H.264 and HEVC streams VideoToolbox decodes, in
//! 8 or 10 bits and 4:2:0, 4:2:2 or 4:4:4, the RGB is byte-equal to what
//! the `ffmpeg` CLI (libswscale, which torchcodec also converts with) produces;
//! see `tests/media_video_exact.rs`.
//!
//! # How the frames are chosen
//!
//! The model's video processor needs the frame count and the rate first
//! (`int(duration * fps)` frames at `int(i * average_fps / fps)`), then a
//! handful of frames. [`VideoSource::open`] therefore reads the track without
//! decoding (an `AVAssetReaderTrackOutput` with no output settings returns the
//! compressed samples): the exact frame count, each frame's presentation time
//! in presentation order, and the track's `preferredTransform`. torchcodec's
//! exact seek mode derives `duration` as the span of the packets (last
//! presentation time plus its duration, minus the first) and `fps` as the frame
//! count over that duration, and [`VideoInfo::fps`] and [`VideoInfo::duration`]
//! are those figures, not the header's `nominalFrameRate` (kept as
//! [`VideoInfo::nominal_fps`]).
//!
//! Decoding still runs through every frame (a P-frame cannot be skipped), but
//! only the sampled frames are converted and kept, so memory is
//! `sampled * width * height * 3`, not the whole video.
//!
//! # Colour
//!
//! AVFoundation's own decode (`AVAssetReader` with output settings) returns the
//! right luma but altered chroma, so it is not used for H.264 and HEVC
//! ([`ColorPath::Swscale`], the default). Their compressed samples go to a
//! `VTDecompressionSession` directly (`vt.rs`) in the stream's own pixel format
//! (8 or 10 bits; 4:2:0, 4:2:2 or 4:4:4), whose planes are byte-equal to FFmpeg's,
//! and [`yuv::to_rgb24`] converts them with libswscale's arithmetic for that format
//! and the matrix and range the stream is tagged with (untagged means BT.601
//! limited, as libswscale does). One class depends on a scaler setting libswscale
//! takes from the caller: 10-bit 4:2:0 (the vertical chroma filter,
//! [`VideoOptions::chroma_filter`], bilinear by default). A stream that cannot be converted exactly (another
//! codec, 12-bit or monochrome video, a matrix [`yuv::Matrix`] does not hold, a
//! shape libswscale does not define) is **refused by name** with
//! [`MediaError::Unsupported`]; only an explicit [`ColorPath::AvFoundation`]
//! converts it, with the system's arithmetic ([`Conversion::AvFoundation`]).
//!
//! # Rotation
//!
//! The frames come out as stored; the track's `preferredTransform` is applied
//! by this module as one of the eight EXIF orientations ([`Rotation::Apply`],
//! the default). The order relative to the colour conversion changes the bytes,
//! and is the `ffmpeg` CLI's autorotate order (measured, `yuv.rs` experiment 9): 4:2:0
//! planes are turned first and converted at the upright size (the converter's choice
//! depends on the width, and the 10-bit chroma filter on the rows, of the upright
//! picture), 4:2:2 and 4:4:4 frames are converted first and the RGB is turned.
//! What torchcodec does with a rotation flag was not run, so its order is
//! unverified. [`Rotation::Ignore`] returns the frames as stored.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use block2::RcBlock;
use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::msg_send;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyClass, AnyObject};
use objc2_foundation::{NSError, NSString, NSURL};

use super::cf::{CFDataGetBytePtr, CFDataGetLength, CFDictionaryGetValue, CFGetTypeID, CfRef, CfTypeRef};
use super::yuv::{self, Chroma, ChromaFilter, Flavor, Layout, Matrix, Params, Picture, Range, Threads};
use super::{codec_config, containers, orient, vt, MediaError};
use crate::embedgemma2_pre::{
    preprocess_video, sample_frame_indices, PreparedVideo, RawImage, VideoConfig, MAX_IMAGE_PIXELS, MAX_VIDEO_FRAMES,
};

// ---------------------------------------------------------------------------
// FFI: CoreMedia (CMTime.h, CMSampleBuffer.h), CoreVideo (CVPixelBuffer.h)
// and the AVFoundation constants; the Objective-C calls go through objc2.
// ---------------------------------------------------------------------------

/// `CMTime` (`CMTime.h`): `value / timescale` seconds when `flags & 1`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct CmTime {
    pub(super) value: i64,
    pub(super) timescale: i32,
    flags: u32,
    epoch: i64,
}

// SAFETY: `CMTime` is the anonymous C struct `{?=qiIq}`, laid out as above.
unsafe impl Encode for CmTime {
    const ENCODING: Encoding = Encoding::Struct(
        "?",
        &[Encoding::LongLong, Encoding::Int, Encoding::UInt, Encoding::LongLong],
    );
}

/// The pointee of a `CMSampleBufferRef` (`struct opaqueCMSampleBuffer`). objc2
/// checks a `msg_send!` return type against the method's encoding in debug
/// builds, and the runtime spells `copyNextSampleBuffer`'s return
/// `^{opaqueCMSampleBuffer=}`; a `*const c_void` (`^v`) passes in release (the
/// ABI is the same) and panics in debug.
#[repr(C)]
struct OpaqueCmSampleBuffer {
    _private: [u8; 0],
}

// SAFETY: a pointer to `struct opaqueCMSampleBuffer` is encoded
// `^{opaqueCMSampleBuffer=}`, which is what the runtime reports for the method.
unsafe impl RefEncode for OpaqueCmSampleBuffer {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("opaqueCMSampleBuffer", &[]));
}

/// `CGAffineTransform` (`CGAffineTransform.h`): six `CGFloat` (`double`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CgAffineTransform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
}

// SAFETY: `CGAffineTransform` is `{CGAffineTransform=dddddd}`.
unsafe impl Encode for CgAffineTransform {
    const ENCODING: Encoding = Encoding::Struct(
        "CGAffineTransform",
        &[
            Encoding::Double,
            Encoding::Double,
            Encoding::Double,
            Encoding::Double,
            Encoding::Double,
            Encoding::Double,
        ],
    );
}

/// `kCMTimeFlags_Valid`.
const CMTIME_VALID: u32 = 1;
/// `kCVPixelFormatType_32BGRA` ('BGRA').
const PIXEL_FORMAT_32BGRA: u32 = u32::from_be_bytes(*b"BGRA");
/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange` ('420v'): 8-bit 4:2:0,
/// Y then interleaved Cb/Cr, luma 16..=235.
const PIXEL_FORMAT_420V: u32 = u32::from_be_bytes(*b"420v");
/// `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange` ('420f'): the same, 0..=255.
const PIXEL_FORMAT_420F: u32 = u32::from_be_bytes(*b"420f");
/// `kCVPixelBufferLock_ReadOnly`.
const PIXEL_BUFFER_LOCK_READ_ONLY: u64 = 1;
/// How far a decoded frame's presentation time may be from the first pass's, in
/// seconds (CMTime timescales are integers, so this is only rounding).
const PTS_TOLERANCE: f64 = 1e-6;
/// `AVAssetReaderStatus` (`AVAssetReader.h`): `Reading` and `Completed`.
const READER_STATUS_READING: isize = 1;
const READER_STATUS_COMPLETED: isize = 2;

#[link(name = "AVFoundation", kind = "framework")]
extern "C" {
    static AVMediaTypeVideo: *const AnyObject;
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    static kCMFormatDescriptionExtension_FullRangeVideo: CfTypeRef;
    static kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms: CfTypeRef;

    fn CMFormatDescriptionGetMediaSubType(desc: CfTypeRef) -> u32;

    fn CMSampleBufferGetNumSamples(sbuf: CfTypeRef) -> isize;
    fn CMSampleBufferGetPresentationTimeStamp(sbuf: CfTypeRef) -> CmTime;
    fn CMSampleBufferGetDuration(sbuf: CfTypeRef) -> CmTime;
    fn CMSampleBufferGetImageBuffer(sbuf: CfTypeRef) -> CfTypeRef;
    fn CMTimeGetSeconds(time: CmTime) -> f64;
    fn CMFormatDescriptionGetExtension(desc: CfTypeRef, key: CfTypeRef) -> CfTypeRef;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFBooleanGetTypeID() -> usize;
    fn CFDataGetTypeID() -> usize;
    fn CFBooleanGetValue(boolean: CfTypeRef) -> u8;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: *const AnyObject;
    /// `kCMFormatDescriptionExtension_YCbCrMatrix` is a macro for this key.
    static kCVImageBufferYCbCrMatrixKey: CfTypeRef;

    fn CVPixelBufferLockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: CfTypeRef) -> *const u8;
    fn CVPixelBufferGetBytesPerRow(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetWidth(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetHeight(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetPixelFormatType(pb: CfTypeRef) -> u32;
    /// The H.273 `matrix_coefficients` code of a `kCVImageBufferYCbCrMatrix_*`
    /// string; 2 (unspecified) for NULL or one it does not know.
    fn CVYCbCrMatrixGetIntegerCodePointForString(matrix: CfTypeRef) -> i32;
}

// ---------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------

/// What to do with the track's `preferredTransform` (rotation metadata).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    /// Rotate or flip the frames to display upright.
    Apply,
    /// Return the frames as stored.
    Ignore,
}

/// How the decoder's YCbCr frames become RGB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorPath {
    /// Decode the compressed samples with VideoToolbox into the decoder's own
    /// planes (see `vt.rs`) and convert them with [`yuv::to_rgb24`], which
    /// reproduces libswscale byte for byte (what torchcodec and the `ffmpeg` CLI
    /// compute), for the given converter flavor. H.264 and HEVC at 8 or 10 bits and
    /// 4:2:0, 4:2:2 or 4:4:4 are exact. A stream that cannot be (see the module
    /// docs) is refused with [`MediaError::Unsupported`], naming why; it is never
    /// converted approximately.
    Swscale(Flavor),
    /// AVFoundation's own decode and YCbCr to BGRA conversion (`AVAssetReader`
    /// and CoreVideo), for any stream it decodes. Its luma equals the decoder's,
    /// but its chroma does not (8 to 15 percent of the chroma samples differ, by up
    /// to about 70, on the fixtures), so this is not libswscale and not close to it
    /// either. Chosen explicitly, never as a fallback.
    AvFoundation,
}

impl Default for ColorPath {
    fn default() -> Self {
        ColorPath::Swscale(Flavor::Aarch64)
    }
}

/// Which conversion produced the frames of a [`DecodedVideo`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conversion {
    /// [`yuv::to_rgb24`] with this layout, matrix, range and flavor (and, for
    /// 10-bit 4:2:0, chroma filter).
    Swscale {
        layout: Layout,
        matrix: Matrix,
        range: Range,
        flavor: Flavor,
        filter: ChromaFilter,
    },
    /// AVFoundation's BGRA conversion.
    AvFoundation,
}

/// Options for opening and decoding a video.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoOptions {
    pub rotation: Rotation,
    /// How decoded YCbCr becomes RGB.
    pub color: ColorPath,
    /// The vertical chroma filter of 10-bit 4:2:0 under [`ColorPath::Swscale`]:
    /// the scaler flag the reference uses (see `yuv.rs`; nothing else depends on it).
    pub chroma_filter: ChromaFilter,
    /// Threads converting a frame's rows (the bytes do not depend on it).
    pub threads: Threads,
    /// The most frames [`VideoSource::decode_frames`] returns.
    pub max_frames: usize,
    /// The most bytes of RGB frames it returns.
    pub max_bytes: usize,
    /// How long to wait for AVFoundation to load the file's tracks.
    pub load_timeout: Duration,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            rotation: Rotation::Apply,
            color: ColorPath::default(),
            chroma_filter: ChromaFilter::default(),
            threads: Threads::default(),
            max_frames: 4096,
            max_bytes: 1 << 32,
            load_timeout: Duration::from_secs(60),
        }
    }
}

/// What the stream says about its own colour, as FFmpeg reads it: the
/// matrix (`matrix_coefficients`, an H.273 code) and range the track's format
/// description carries. `None` where the stream does not say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamTags {
    pub matrix_code: Option<u8>,
    pub full_range: Option<bool>,
}

/// What the first (undecoded) pass found.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    /// The exact number of frames in the video track.
    pub total_frames: usize,
    /// The frame rate the reference derives from the track
    /// (torchcodec's `average_fps` in its exact seek mode, which
    /// `transformers` asks for): `total_frames / duration`, in `f64`.
    pub fps: f64,
    /// `max(pts + duration) - min(pts)` over the frames, in seconds
    /// (torchcodec's `duration_seconds` in exact mode).
    pub duration: f64,
    /// The track's `nominalFrameRate`, which is what the header says.
    pub nominal_fps: f64,
    /// Each frame's presentation time in seconds, in presentation order.
    pub timestamps: Vec<f64>,
    /// The EXIF orientation (1 to 8) the track's `preferredTransform` stands
    /// for; 1 when it is the identity.
    pub orientation: u32,
    /// The colour tags the stream carries.
    pub tags: StreamTags,
    /// The sample layout (depth and chroma subsampling) the codec configuration
    /// declares for an H.264 or HEVC stream, the streams [`ColorPath::Swscale`]
    /// converts exactly; otherwise why this stream has none (the reason
    /// [`ColorPath::Swscale`] refuses it with).
    pub layout: Result<Layout, String>,
    /// The size of the stored frames, before rotation.
    pub stored_width: usize,
    pub stored_height: usize,
}

/// Decoded frames and where they came from.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedVideo {
    /// Width of every frame (after rotation, if applied).
    pub width: usize,
    pub height: usize,
    /// The reference's average frame rate ([`VideoInfo::fps`]).
    pub fps: f64,
    /// Frames in the whole video.
    pub total_frames: usize,
    /// Source index (presentation order) of each returned frame.
    pub indices: Vec<usize>,
    /// Presentation time of each returned frame, seconds.
    pub timestamps: Vec<f64>,
    /// One `height * width * 3` RGB buffer per returned frame.
    pub frames: Vec<Vec<u8>>,
    /// How the frames were converted to RGB.
    pub conversion: Conversion,
}

/// One plane's samples: bytes for 8-bit pictures, 16-bit values (right-aligned,
/// as FFmpeg's `yuv420p10le` holds them) for deeper ones.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Samples {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl Samples {
    /// The number of samples.
    pub fn len(&self) -> usize {
        match self {
            Samples::U8(v) => v.len(),
            Samples::U16(v) => v.len(),
        }
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A decoded frame's planes as VideoToolbox produced them, tightly packed: a
/// `width * height` luma plane, then interleaved Cb/Cr pairs, `ceil(height / sub)`
/// rows of `2 * ceil(width / sub)` samples with the layout's subsampling.
/// Not rotated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeFrame {
    pub width: usize,
    pub height: usize,
    /// Bit depth and chroma subsampling.
    pub layout: Layout,
    /// Whether the buffer is a full-range format (`420f`, `xf20`, ...).
    pub full_range: bool,
    pub y: Samples,
    pub uv: Samples,
}

impl DecodedVideo {
    /// The frames as the preprocessing module's input.
    pub fn raw_frames(&self) -> Vec<RawImage<'_>> {
        self.frames
            .iter()
            .map(|f| RawImage::rgb(self.width, self.height, f))
            .collect()
    }
}

/// An opened video: its track and the facts of the first pass.
pub struct VideoSource {
    asset: Retained<AnyObject>,
    track: Retained<AnyObject>,
    info: VideoInfo,
    options: VideoOptions,
    /// The frames' presentation times in media ticks, in presentation order: how a
    /// decoded sample is told from its time which frame it is.
    media_pts: Vec<i64>,
    /// A temporary file made from bytes, removed on drop.
    _temp: Option<TempFile>,
}

impl std::fmt::Debug for VideoSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoSource")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

/// A file in the temp directory, created exclusively and removed on drop.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        // Best effort: the file is private scratch space.
        let _ = std::fs::remove_file(&self.0);
    }
}

impl VideoSource {
    /// Open a video file and run the first pass.
    ///
    /// # Errors
    /// [`MediaError::Io`] when the file cannot be read, [`MediaError::Empty`]
    /// for an empty one, [`MediaError::Unrecognized`] when it has no video
    /// track, [`MediaError::Decode`] when the track cannot be read through (a
    /// truncated or damaged file), [`MediaError::Unsupported`] for a track
    /// whose samples hold several frames or whose transform is not a right
    /// angle (with [`Rotation::Apply`]).
    pub fn open(path: &Path, options: &VideoOptions) -> Result<VideoSource, MediaError> {
        open_inner(path, options, None)
    }

    /// Open a video from its bytes (written to a private temporary file for
    /// AVFoundation, which reads files). `extension` ("mp4", "mov", "m4v") picks
    /// the type AVFoundation sniffs it as.
    ///
    /// # Errors
    /// As [`VideoSource::open`], plus [`MediaError::Invalid`] for an unusable
    /// extension.
    pub fn open_bytes(bytes: &[u8], extension: &str, options: &VideoOptions) -> Result<VideoSource, MediaError> {
        if bytes.is_empty() {
            return Err(MediaError::Empty { what: "video" });
        }
        if extension.is_empty() || extension.len() > 8 || !extension.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(MediaError::Invalid {
                what: "video",
                detail: format!("extension {extension:?} must be 1 to 8 letters or digits"),
            });
        }
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "tessl-media-{}-{}-{nanos}.{extension}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| MediaError::Io {
                path: path.clone(),
                source,
            })?;
        let temp = TempFile(path.clone());
        file.write_all(bytes).map_err(|source| MediaError::Io {
            path: path.clone(),
            source,
        })?;
        drop(file);
        open_inner(&path, options, Some(temp))
    }

    /// The first pass's findings.
    pub fn info(&self) -> &VideoInfo {
        &self.info
    }

    /// Decode the frames at `indices` (presentation order, duplicates allowed),
    /// returned in the order asked.
    ///
    /// # Errors
    /// [`MediaError::Invalid`] for an index past the last frame or an empty
    /// list, [`MediaError::Limit`] over `max_frames`, `max_bytes` or the pixel
    /// bound, [`MediaError::Decode`] when AVFoundation fails mid-stream or
    /// decodes a different number of frames than the first pass counted.
    pub fn decode_frames(&self, indices: &[usize]) -> Result<DecodedVideo, MediaError> {
        let wanted = self.checked_wanted(indices)?;
        autoreleasepool(|_| self.decode_wanted(&wanted, indices))
    }

    /// Decode the frames the model's video processor samples
    /// ([`VideoSource::sample_indices`]); with `cfg.do_sample_frames` false,
    /// every frame (up to `max_frames`).
    ///
    /// # Errors
    /// As [`VideoSource::decode_frames`], and the sampler's own errors.
    pub fn decode_sampled(&self, cfg: &VideoConfig) -> Result<DecodedVideo, MediaError> {
        let indices = if cfg.do_sample_frames {
            self.sample_indices(cfg)?
        } else {
            if self.info.total_frames > MAX_VIDEO_FRAMES {
                return Err(MediaError::Limit {
                    what: "video",
                    detail: format!("{} frames exceeds {MAX_VIDEO_FRAMES}", self.info.total_frames),
                });
            }
            (0..self.info.total_frames).collect()
        };
        self.decode_frames(&indices)
    }

    /// The frame indices `EmbeddingGemma2VideoProcessor.sample_frames` picks for
    /// this video: with a target `fps`, `max(1, int(duration * fps))` frames at
    /// `min(total - 1, int(i * source_fps / fps))`, where `duration` and
    /// `source_fps` are the metadata torchcodec reports ([`VideoInfo::duration`]
    /// and [`VideoInfo::fps`], which are not `total / fps` to the last bit);
    /// then the `max_frames` cap with the overflow strategy.
    ///
    /// # Errors
    /// [`MediaError::Preprocess`] with the sampler's message.
    pub fn sample_indices(&self, cfg: &VideoConfig) -> Result<Vec<usize>, MediaError> {
        let total = self.info.total_frames;
        let mut indices: Vec<usize> = match cfg.fps {
            None => (0..total).collect(),
            Some(fps) => {
                let step = self.info.fps / fps;
                let sampled = ((self.info.duration * fps) as usize).max(1);
                (0..sampled)
                    .map(|i| ((i as f64 * step) as usize).min(total - 1))
                    .collect()
            }
        };
        if indices.len() > cfg.max_frames {
            // The cap chooses positions in the list, as it would in `arange(n)`.
            let mut cap = cfg.clone();
            cap.fps = None;
            let positions = sample_frame_indices(&cap, indices.len(), 1.0).map_err(MediaError::Preprocess)?;
            indices = positions.into_iter().map(|p| indices[p]).collect();
        }
        Ok(indices)
    }

    /// Decode the frames at `indices` to the decoder's own planes, without
    /// converting or rotating them (for checking the planes against another
    /// decoder's). They come from VideoToolbox driven directly (see `vt.rs` for
    /// why not from `AVAssetReader`'s decoded output).
    ///
    /// # Errors
    /// As [`VideoSource::decode_frames`], and [`MediaError::Unsupported`] when
    /// the track is not an H.264 or HEVC stream of a layout VideoToolbox decodes
    /// to one of the formats [`vt`] knows (8 or 10 bits; 4:2:0, 4:2:2, 4:4:4).
    pub fn decode_native_planes(&self, indices: &[usize]) -> Result<Vec<NativeFrame>, MediaError> {
        let wanted = self.checked_wanted(indices)?;
        let layout = self.exact_layout()?;
        let mut kept = autoreleasepool(|_| self.read_native(&wanted, layout))?;
        indices
            .iter()
            .map(|i| {
                wanted
                    .binary_search(i)
                    .ok()
                    .and_then(|at| kept.get_mut(at))
                    .map(|slot| slot.clone())
                    .ok_or_else(|| MediaError::Decode {
                        what: "video",
                        detail: "a requested frame was not decoded".into(),
                    })
            })
            .collect()
    }

    /// Decode the whole track with VideoToolbox and return the planes of the
    /// sorted, unique `wanted` frames, in that order.
    fn read_native(&self, wanted: &[usize], layout: Layout) -> Result<Vec<NativeFrame>, MediaError> {
        const WHAT: &str = "video";
        let full_range = self.info.tags.full_range == Some(true);
        let total = self.info.total_frames;
        let mut flags = vec![false; total];
        for &w in wanted {
            flags[w] = true;
        }
        // Compressed samples, in decode order, with the track's format description.
        let (reader, output) = new_reader(&self.asset, &self.track, None)?;
        start_reading(&reader)?;
        let mut decoder: Option<vt::Decoder> = None;
        let mut samples = 0usize;
        loop {
            let done = autoreleasepool(|_| -> Result<bool, MediaError> {
                let Some(sample) = next_sample(&output) else {
                    return Ok(true);
                };
                // SAFETY: `sample` is a live CMSampleBuffer.
                if unsafe { CMSampleBufferGetNumSamples(sample.as_ptr()) } == 0 {
                    return Ok(false); // a marker-only buffer
                }
                if decoder.is_none() {
                    decoder = Some(vt::Decoder::new(
                        sample.as_ptr(),
                        layout,
                        full_range,
                        self.media_pts.clone(),
                        flags.clone(),
                    )?);
                }
                if let Some(d) = decoder.as_mut() {
                    d.decode(sample.as_ptr())?;
                }
                samples += 1;
                Ok(false)
            })?;
            if done {
                break;
            }
        }
        if reader.status() != READER_STATUS_COMPLETED {
            return Err(MediaError::Decode {
                what: WHAT,
                detail: format!(
                    "AVAssetReader stopped after {samples} of {total} samples ({})",
                    reader.error_text()
                ),
            });
        }
        if samples != total {
            return Err(MediaError::Decode {
                what: WHAT,
                detail: format!("read {samples} samples where the first pass counted {total}"),
            });
        }
        let frames = decoder
            .ok_or_else(|| MediaError::Decode {
                what: WHAT,
                detail: "the track has no samples".into(),
            })?
            .finish()?;
        wanted
            .iter()
            .map(|&i| {
                frames.get(i).cloned().flatten().ok_or_else(|| MediaError::Decode {
                    what: WHAT,
                    detail: format!("frame {i} was not decoded"),
                })
            })
            .collect()
    }

    /// The stream's sample layout, or why it has none that is converted exactly.
    fn exact_layout(&self) -> Result<Layout, MediaError> {
        self.info.layout.clone().map_err(|reason| MediaError::Unsupported {
            what: "video",
            detail: format!(
                "{reason}; the colour conversion is exact only for H.264 and HEVC at 8 or 10 bits in 4:2:0, 4:2:2 \
                 or 4:4:4 (ColorPath::AvFoundation converts anything the system decodes, with its own arithmetic)"
            ),
        })
    }

    /// The conversion this source's frames get under its options, or the
    /// reason there is no exact one.
    fn plan(&self) -> Result<Plan, MediaError> {
        let ColorPath::Swscale(flavor) = self.options.color else {
            return Ok(Plan::AvFoundation);
        };
        let info = &self.info;
        let layout = self.exact_layout()?;
        let range = if info.tags.full_range == Some(true) {
            Range::Full
        } else {
            Range::Limited
        };
        // A stream that does not name its matrix is converted as libswscale
        // does: with BT.601's, whatever its size.
        let matrix = match info.tags.matrix_code {
            None => Matrix::Bt601,
            Some(code) => Matrix::from_code(code).ok_or_else(|| MediaError::Unsupported {
                what: "video",
                detail: format!(
                    "the stream's matrix_coefficients {code} is not one the exact conversion holds (BT.709, BT.601, \
                     BT.2020 non-constant, FCC, SMPTE 240M); ColorPath::AvFoundation converts it with the system's arithmetic"
                ),
            })?,
        };
        let filter = self.options.chroma_filter;
        // 4:2:0 is converted after it is turned (see `convert_native`), so its shape is the upright one.
        let (w, h) = if layout.chroma == Chroma::C420 && self.options.rotation == Rotation::Apply {
            orient::oriented_size(info.orientation, info.stored_width, info.stored_height)
        } else {
            (info.stored_width, info.stored_height)
        };
        yuv::check_shape(layout, w, h, filter).map_err(|e| MediaError::Unsupported {
            what: "video",
            detail: e.to_string(),
        })?;
        Ok(Plan::Swscale {
            layout,
            matrix,
            range,
            flavor,
            filter,
        })
    }

    fn checked_wanted(&self, indices: &[usize]) -> Result<Vec<usize>, MediaError> {
        if indices.is_empty() {
            return Err(MediaError::Invalid {
                what: "video",
                detail: "no frame indices asked for".into(),
            });
        }
        if indices.len() > self.options.max_frames {
            return Err(MediaError::Limit {
                what: "video",
                detail: format!("{} frames exceeds {}", indices.len(), self.options.max_frames),
            });
        }
        if let Some(&bad) = indices.iter().find(|&&i| i >= self.info.total_frames) {
            return Err(MediaError::Invalid {
                what: "video",
                detail: format!("frame {bad} asked for, the video has {}", self.info.total_frames),
            });
        }
        let mut wanted: Vec<usize> = indices.to_vec();
        wanted.sort_unstable();
        wanted.dedup();
        Ok(wanted)
    }

    /// Decode the whole track with `settings`, handing the pixel buffer of each
    /// frame in the sorted, unique `wanted` to `visit`, and check the decode
    /// against the first pass (every frame's time, the frame count, the
    /// reader's status).
    fn read_frames<T>(
        &self,
        settings: &Retained<AnyObject>,
        wanted: &[usize],
        mut visit: impl FnMut(CfTypeRef) -> Result<T, MediaError>,
    ) -> Result<Vec<T>, MediaError> {
        const WHAT: &str = "video";
        let (reader, output) = new_reader(&self.asset, &self.track, Some(settings))?;
        start_reading(&reader)?;
        let mut kept: Vec<T> = Vec::with_capacity(wanted.len());
        let mut next = 0usize;
        let mut index = 0usize;
        loop {
            let done = autoreleasepool(|_| -> Result<bool, MediaError> {
                let Some(sample) = next_sample(&output) else {
                    return Ok(true);
                };
                // SAFETY: `sample` is a live CMSampleBuffer.
                let image = unsafe { CMSampleBufferGetImageBuffer(sample.as_ptr()) };
                if image.is_null() {
                    return Ok(false); // a buffer without a picture is not a frame
                }
                // SAFETY: `sample` is a live CMSampleBuffer; the time is a value.
                let decoded_pts = unsafe { CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample.as_ptr())) };
                match self.info.timestamps.get(index) {
                    Some(&first_pass) if (decoded_pts - first_pass).abs() <= PTS_TOLERANCE => {}
                    other => {
                        return Err(MediaError::Decode {
                            what: WHAT,
                            detail: format!(
                                "frame {index} decodes at {decoded_pts} s where the first pass found {other:?}: the track's edit list drops or reorders frames"
                            ),
                        })
                    }
                }
                if next < wanted.len() && wanted[next] == index {
                    kept.push(visit(image)?);
                    next += 1;
                }
                index += 1;
                Ok(false)
            })?;
            if done {
                break;
            }
        }
        // The loop ends when `copyNextSampleBuffer` returns NULL, which is the end
        // of the track or a failure; only the reader's status tells which.
        if reader.status() != READER_STATUS_COMPLETED {
            return Err(MediaError::Decode {
                what: WHAT,
                detail: format!(
                    "AVAssetReader stopped after {index} of {} frames ({})",
                    self.info.total_frames,
                    reader.error_text()
                ),
            });
        }
        if index != self.info.total_frames || next != wanted.len() {
            return Err(MediaError::Decode {
                what: WHAT,
                detail: format!(
                    "decoded {index} frames where the first pass counted {}",
                    self.info.total_frames
                ),
            });
        }
        Ok(kept)
    }

    /// Pass two: decode and convert the sorted, unique `wanted` frames, then lay
    /// them out in the order of `asked`.
    fn decode_wanted(&self, wanted: &[usize], asked: &[usize]) -> Result<DecodedVideo, MediaError> {
        const WHAT: &str = "video";
        let plan = self.plan()?;
        let mut dims: Option<(usize, usize)> = None;
        // Checks each converted frame against the first one's size and the byte bound.
        let mut emit = |convert: &dyn Fn(Option<(usize, usize)>) -> Converted| {
            let (rgb, w, h) = convert(dims)?;
            if dims.is_none() {
                let per = w * h * 3;
                let total = per.checked_mul(wanted.len());
                if total.is_none_or(|t| t > self.options.max_bytes) {
                    return Err(MediaError::Limit {
                        what: WHAT,
                        detail: format!(
                            "{} frames of {w}x{h} exceed {} bytes",
                            wanted.len(),
                            self.options.max_bytes
                        ),
                    });
                }
                dims = Some((w, h));
            }
            Ok(rgb)
        };
        let (orientation, rotation) = (self.info.orientation, self.options.rotation);
        let kept: Vec<Vec<u8>> = match plan {
            Plan::Swscale {
                layout,
                matrix,
                range,
                flavor,
                filter,
            } => {
                let params = Params {
                    matrix,
                    range,
                    flavor,
                    filter,
                    threads: self.options.threads,
                };
                self.read_native(wanted, layout)?
                    .iter()
                    .map(|f| emit(&|expect| convert_native(f, &params, orientation, rotation, expect)))
                    .collect::<Result<_, _>>()?
            }
            Plan::AvFoundation => {
                let settings = bgra_settings()?;
                self.read_frames(&settings, wanted, |image| {
                    emit(&|expect| convert_frame(image, orientation, rotation, expect))
                })?
            }
        };
        let (width, height) = dims.ok_or_else(|| MediaError::Decode {
            what: WHAT,
            detail: "no frame was converted".into(),
        })?;
        // Hand each frame out once; a frame asked for twice is cloned.
        let mut uses = vec![0usize; wanted.len()];
        let slots: Vec<usize> = asked
            .iter()
            .map(|i| {
                let at = wanted.binary_search(i).unwrap_or(0);
                uses[at] += 1;
                at
            })
            .collect();
        let mut kept: Vec<Option<Vec<u8>>> = kept.into_iter().map(Some).collect();
        let mut frames = Vec::with_capacity(asked.len());
        for at in slots {
            uses[at] -= 1;
            let frame = if uses[at] == 0 {
                kept[at].take()
            } else {
                kept[at].clone()
            };
            frames.push(frame.ok_or_else(|| MediaError::Decode {
                what: WHAT,
                detail: "a requested frame was not decoded".into(),
            })?);
        }
        Ok(DecodedVideo {
            width,
            height,
            fps: self.info.fps,
            total_frames: self.info.total_frames,
            indices: asked.to_vec(),
            timestamps: asked.iter().map(|&i| self.info.timestamps[i]).collect(),
            frames,
            conversion: match plan {
                Plan::Swscale {
                    layout,
                    matrix,
                    range,
                    flavor,
                    filter,
                } => Conversion::Swscale {
                    layout,
                    matrix,
                    range,
                    flavor,
                    filter,
                },
                Plan::AvFoundation => Conversion::AvFoundation,
            },
        })
    }
}

/// The conversion [`VideoSource`] applies, decided once per source.
#[derive(Clone, Copy, Debug)]
enum Plan {
    Swscale {
        layout: Layout,
        matrix: Matrix,
        range: Range,
        flavor: Flavor,
        filter: ChromaFilter,
    },
    AvFoundation,
}

/// Decode the sampled frames of a video file, in one call.
///
/// # Errors
/// As [`VideoSource::open`] and [`VideoSource::decode_sampled`].
pub fn decode_video_file(path: &Path, cfg: &VideoConfig, options: &VideoOptions) -> Result<DecodedVideo, MediaError> {
    VideoSource::open(path, options)?.decode_sampled(cfg)
}

/// Preprocess frames already sampled by [`VideoSource::decode_sampled`]: the
/// per-frame image path of [`preprocess_video`] with its own sampling off, so
/// the frames are not sampled twice, and the true source indices and
/// timestamps (`index / fps`, the processor's convention) restored.
///
/// # Errors
/// [`MediaError::Preprocess`] with the preprocessing module's message.
pub fn preprocess_decoded_video(cfg: &VideoConfig, video: &DecodedVideo) -> Result<PreparedVideo, MediaError> {
    let mut cfg = cfg.clone();
    cfg.do_sample_frames = false;
    let frames = video.raw_frames();
    let mut out = preprocess_video(&cfg, &frames, video.fps).map_err(MediaError::Preprocess)?;
    out.indices = video.indices.clone();
    out.timestamps = video.indices.iter().map(|&i| i as f64 / video.fps).collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// AVFoundation plumbing
// ---------------------------------------------------------------------------

fn framework(api: &'static str, detail: impl Into<String>) -> MediaError {
    MediaError::Framework {
        api,
        detail: detail.into(),
    }
}

/// `-[NSError localizedDescription]`, or a placeholder for nil.
fn ns_error_text(error: Option<&NSError>) -> String {
    match error {
        Some(e) => e.localizedDescription().to_string(),
        None => "no error reported".to_string(),
    }
}

/// An `AVAssetReader` that is cancelled if it is dropped mid-read.
struct Reader(Retained<AnyObject>);

impl Reader {
    /// `AVAssetReaderStatus`.
    fn status(&self) -> isize {
        // SAFETY: `self.0` is a live AVAssetReader; `status` is a thread-safe
        // read-only property returning an NSInteger.
        unsafe { msg_send![&*self.0, status] }
    }

    /// The reader's `error` as text.
    fn error_text(&self) -> String {
        // SAFETY: as above; `error` is nil or an NSError.
        let error: Option<Retained<NSError>> = unsafe { msg_send![&*self.0, error] };
        ns_error_text(error.as_deref())
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        if self.status() == READER_STATUS_READING {
            // SAFETY: `cancelReading` is valid on a reader that is reading, and
            // releases the decoder's resources before the reader is released.
            let () = unsafe { msg_send![&*self.0, cancelReading] };
        }
    }
}

fn class(name: &'static std::ffi::CStr) -> Result<&'static AnyClass, MediaError> {
    AnyClass::get(name).ok_or_else(|| framework("objc_getClass", format!("{name:?} is not available")))
}

fn open_inner(path: &Path, options: &VideoOptions, temp: Option<TempFile>) -> Result<VideoSource, MediaError> {
    autoreleasepool(|_| {
        let meta = std::fs::metadata(path).map_err(|source| MediaError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if !meta.is_file() {
            return Err(MediaError::Invalid {
                what: "video",
                detail: format!("{} is not a file", path.display()),
            });
        }
        if meta.len() == 0 {
            return Err(MediaError::Empty { what: "video" });
        }
        // AVFoundation reads the frames a cut-short file still holds without
        // complaint, so the boxes' declared sizes are compared with the file.
        let source = containers::FileSource {
            file: std::fs::File::open(path).map_err(|source| MediaError::Io {
                path: path.to_path_buf(),
                source,
            })?,
            len: meta.len(),
        };
        containers::check_video(&source).map_err(|detail| MediaError::Decode { what: "video", detail })?;
        let path_str = path.to_str().ok_or_else(|| MediaError::Invalid {
            what: "video",
            detail: "the path is not UTF-8".into(),
        })?;
        let ns_path = NSString::from_str(path_str);
        let url = NSURL::fileURLWithPath(&ns_path);
        // SAFETY: `+[AVURLAsset URLAssetWithURL:options:]` takes an NSURL and a
        // nullable options dictionary and returns an autoreleased AVURLAsset.
        let asset: Option<Retained<AnyObject>> =
            unsafe { msg_send![class(c"AVURLAsset")?, URLAssetWithURL: &*url, options: std::ptr::null::<AnyObject>()] };
        let asset = asset.ok_or_else(|| framework("AVURLAsset", "no asset for the file"))?;
        let track = first_video_track(&asset, options.load_timeout)?;
        let track_frame_rate: f32 = {
            // SAFETY: `track` is a live AVAssetTrack; `nominalFrameRate` is a
            // float property (the header marks it deprecated for Swift only;
            // Objective-C reads it, blocking until it is loaded).
            unsafe { msg_send![&*track, nominalFrameRate] }
        };
        let transform: CgAffineTransform = {
            // SAFETY: as above, for the `CGAffineTransform` property.
            unsafe { msg_send![&*track, preferredTransform] }
        };
        let orientation = match orientation_of(&transform) {
            Some(o) => o,
            None if options.rotation == Rotation::Ignore => 1,
            None => {
                return Err(MediaError::Unsupported {
                    what: "video",
                    detail: format!(
                        "the track's transform [{} {} {} {}] is not a rotation or a flip",
                        transform.a, transform.b, transform.c, transform.d
                    ),
                })
            }
        };
        let (tags, layout) = stream_format(&track);
        let first_pass = count_frames(&asset, &track)?;
        // Compressed samples carry media time; decoded frames carry track time,
        // with the edit list applied (an H.264 stream with B-frames starts its
        // media at +2 frames and the edit list shifts it back to 0). Report
        // track time, which is what FFmpeg and OpenCV report, and which
        // `decode_wanted` checks against every decoded frame.
        let first = first_decoded(&asset, &track, tags, &layout)?;
        let timescale = f64::from(first_pass.timescale);
        let shift_seconds = first_pass.pts[0] as f64 / timescale - first.seconds;
        let shift = (shift_seconds * timescale).round();
        if (shift_seconds * timescale - shift).abs() > 1e-3 {
            return Err(MediaError::Decode {
                what: "video",
                detail: format!(
                    "the edit list moves the first frame by {shift_seconds} s, not a whole number of media ticks"
                ),
            });
        }
        let shift = shift as i64;
        let pts: Vec<i64> = first_pass.pts.iter().map(|&p| p - shift).collect();
        // torchcodec's exact seek mode (`Metadata.cpp`, `transformers` asks for
        // it): the duration is the span from the earliest presentation time to
        // the latest end (presentation time plus the frame's duration), in
        // seconds as `ticks / timescale`, and the rate is frames over duration.
        let begin = pts[0] as f64 / timescale;
        let end = pts
            .iter()
            .zip(&first_pass.durations)
            .map(|(&p, &d)| p + d)
            .max()
            .map_or(begin, |t| t as f64 / timescale);
        let duration = end - begin;
        let total_frames = pts.len();
        if !(duration.is_finite() && duration > 0.0) {
            return Err(MediaError::Decode {
                what: "video",
                detail: format!("the {total_frames} frames span {duration} s, so the video has no frame rate"),
            });
        }
        let timestamps: Vec<f64> = pts.iter().map(|&p| p as f64 / timescale).collect();
        Ok(VideoSource {
            asset,
            track,
            info: VideoInfo {
                total_frames,
                fps: total_frames as f64 / duration,
                duration,
                nominal_fps: f64::from(track_frame_rate),
                timestamps,
                orientation,
                tags,
                layout,
                stored_width: first.width,
                stored_height: first.height,
            },
            options: *options,
            media_pts: first_pass.pts,
            _temp: temp,
        })
    })
}

/// The EXIF orientation a `preferredTransform` stands for. A pure rotation
/// that is not a right angle is rounded to the nearest multiple of 90 degrees,
/// as torchcodec does (`rotation_from_degrees`); a flip must be exact.
fn orientation_of(t: &CgAffineTransform) -> Option<u32> {
    if let Some(o) = orient::orientation_of_transform(t.a, t.b, t.c, t.d) {
        return Some(o);
    }
    // A rotation matrix [a b; -b a] scaled by s: angle = atan2(b, a), in
    // degrees clockwise on screen (y down); the display matrix's
    // counter-clockwise angle is its negative.
    let scale = t.a.hypot(t.b);
    let rotation_like = scale > 1e-3 && (t.a - t.d).abs() < 1e-3 * scale && (t.b + t.c).abs() < 1e-3 * scale;
    if !rotation_like {
        return None;
    }
    let ccw = -t.b.atan2(t.a).to_degrees();
    match (ccw / 90.0).round().rem_euclid(4.0) as i64 {
        0 => Some(1),
        1 => Some(8),
        2 => Some(3),
        _ => Some(6),
    }
}

/// The colour tags of the track's first format description, and the sample
/// layout of the stream if it is one [`yuv`] converts as libswscale does.
///
/// The bit depth and chroma format are read from the codec configuration the
/// sample description carries (`avcC` for H.264, `hvcC` for HEVC), because the
/// pixel format a decoder is *asked* for says nothing about the stream: a 10-bit
/// stream asked for `420v` is converted to it by the system without complaint.
fn stream_format(track: &Retained<AnyObject>) -> (StreamTags, Result<Layout, String>) {
    let none = StreamTags {
        matrix_code: None,
        full_range: None,
    };
    let missing = || Err("the track has no format description".to_string());
    // SAFETY: `formatDescriptions` is an NSArray of CMFormatDescriptionRef (nil
    // if the track has none); `count` and `objectAtIndex:` are plain reads, and
    // the element stays alive as long as the track does.
    unsafe {
        let descriptions: Option<Retained<AnyObject>> = msg_send![&**track, formatDescriptions];
        let Some(descriptions) = descriptions else {
            return (none, missing());
        };
        let count: usize = msg_send![&*descriptions, count];
        if count == 0 {
            return (none, missing());
        }
        let desc: *const AnyObject = msg_send![&*descriptions, objectAtIndex: 0usize];
        if desc.is_null() {
            return (none, missing());
        }
        let desc: CfTypeRef = desc.cast();
        let matrix = CMFormatDescriptionGetExtension(desc, kCVImageBufferYCbCrMatrixKey);
        let matrix_code = if matrix.is_null() {
            None
        } else {
            u8::try_from(CVYCbCrMatrixGetIntegerCodePointForString(matrix)).ok()
        };
        let range = CMFormatDescriptionGetExtension(desc, kCMFormatDescriptionExtension_FullRangeVideo);
        let full_range = if !range.is_null() && CFGetTypeID(range) == CFBooleanGetTypeID() {
            Some(CFBooleanGetValue(range) != 0)
        } else {
            None
        };
        (
            StreamTags {
                matrix_code,
                full_range,
            },
            stream_layout(desc),
        )
    }
}

/// The codec configuration record (`avcC` or `hvcC`) in a format description's
/// sample description extension atoms, as bytes.
///
/// # Safety
/// `desc` must be a live `CMFormatDescriptionRef`.
unsafe fn configuration_record(desc: CfTypeRef, atom: &str) -> Option<Vec<u8>> {
    let atoms = CMFormatDescriptionGetExtension(desc, kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms);
    if atoms.is_null() {
        return None;
    }
    // The key is a CFString; an NSString is one (toll-free bridged).
    let key = NSString::from_str(atom);
    let data = CFDictionaryGetValue(atoms, (&*key as *const NSString).cast());
    if data.is_null() || CFGetTypeID(data) != CFDataGetTypeID() {
        return None;
    }
    let len = usize::try_from(CFDataGetLength(data)).ok()?;
    let ptr = CFDataGetBytePtr(data);
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `data` is a live CFData of `len` bytes, owned by the dictionary.
    Some(std::slice::from_raw_parts(ptr, len).to_vec())
}

/// The sample layout of the format description's stream: H.264 and HEVC from their
/// configuration record ([`codec_config`]), anything else is why it has none.
///
/// # Safety
/// `desc` must be a live `CMFormatDescriptionRef`.
unsafe fn stream_layout(desc: CfTypeRef) -> Result<Layout, String> {
    let subtype = CMFormatDescriptionGetMediaSubType(desc);
    let is = |code: &[u8; 4]| subtype == u32::from_be_bytes(*code);
    if is(b"avc1") || is(b"avc3") {
        let record = configuration_record(desc, "avcC").ok_or("the H.264 track has no avcC record")?;
        return codec_config::avc_layout(&record);
    }
    if is(b"hvc1") || is(b"hev1") {
        let record = configuration_record(desc, "hvcC").ok_or("the HEVC track has no hvcC record")?;
        return codec_config::hevc_layout(&record);
    }
    let name: String = subtype
        .to_be_bytes()
        .iter()
        .map(|&b| if b.is_ascii_graphic() { char::from(b) } else { '?' })
        .collect();
    Err(format!(
        "the codec '{name}' is not H.264 or HEVC, the codecs whose decode is held to FFmpeg's"
    ))
}

/// A value handed from AVFoundation's completion queue to the caller.
struct Handoff<T>(T);
// SAFETY: the only value sent is a retained NSArray of AVAssetTrack (or a
// string). AVAsset and AVAssetTrack are documented as immutable and
// thread-safe, and Objective-C reference counting is atomic.
unsafe impl<T> Send for Handoff<T> {}

type TrackSlot = Arc<(Mutex<Option<Result<Handoff<Retained<AnyObject>>, String>>>, Condvar)>;

/// The first video track of `asset`, via `loadTracksWithMediaType:completionHandler:`
/// (`-tracksWithMediaType:` is deprecated since macOS 15).
fn first_video_track(asset: &Retained<AnyObject>, timeout: Duration) -> Result<Retained<AnyObject>, MediaError> {
    let slot: TrackSlot = Arc::new((Mutex::new(None), Condvar::new()));
    let sink = Arc::clone(&slot);
    let handler = RcBlock::new(move |tracks: *mut AnyObject, error: *mut AnyObject| {
        let result = if !tracks.is_null() {
            // SAFETY: `tracks` is a live NSArray for the duration of the
            // callback; `retain` takes our own reference.
            match unsafe { Retained::retain(tracks) } {
                Some(t) => Ok(Handoff(t)),
                None => Err("the track list could not be retained".to_string()),
            }
        } else {
            // SAFETY: `error` is nil or a live NSError for the callback.
            let text = unsafe { error.cast::<NSError>().as_ref() }.map(|e| e.localizedDescription().to_string());
            Err(text.unwrap_or_else(|| "no tracks and no error".to_string()))
        };
        let (lock, cv) = &*sink;
        if let Ok(mut guard) = lock.lock() {
            *guard = Some(result);
            cv.notify_all();
        }
    });
    // SAFETY: `AVMediaTypeVideo` is an SDK constant NSString; the selector takes
    // the media type and a block of `(NSArray *, NSError *)`, which `handler` is.
    let () = unsafe {
        msg_send![
            &**asset,
            loadTracksWithMediaType: &*AVMediaTypeVideo,
            completionHandler: &*handler
        ]
    };
    let (lock, cv) = &*slot;
    let guard = lock
        .lock()
        .map_err(|_| framework("loadTracksWithMediaType", "the result lock was poisoned"))?;
    let (mut guard, wait) = cv
        .wait_timeout_while(guard, timeout, |r| r.is_none())
        .map_err(|_| framework("loadTracksWithMediaType", "the result lock was poisoned"))?;
    if wait.timed_out() && guard.is_none() {
        return Err(framework(
            "loadTracksWithMediaType",
            format!("AVFoundation did not load the tracks within {timeout:?}"),
        ));
    }
    let tracks = match guard.take() {
        Some(Ok(Handoff(tracks))) => tracks,
        Some(Err(text)) => {
            return Err(MediaError::unrecognized("video", text));
        }
        None => return Err(framework("loadTracksWithMediaType", "no result")),
    };
    // SAFETY: `tracks` is an NSArray; `count` and `firstObject` are plain reads.
    let count: usize = unsafe { msg_send![&*tracks, count] };
    if count == 0 {
        return Err(MediaError::unrecognized("video", "the file has no video track"));
    }
    // SAFETY: `count > 0`, so `firstObject` is a live AVAssetTrack.
    let track: Option<Retained<AnyObject>> = unsafe { msg_send![&*tracks, firstObject] };
    track.ok_or_else(|| framework("NSArray", "firstObject was nil"))
}

/// A reader over `track` with `settings` (nil: compressed samples) and its
/// single output, not yet started.
fn new_reader(
    asset: &Retained<AnyObject>,
    track: &Retained<AnyObject>,
    settings: Option<&AnyObject>,
) -> Result<(Reader, Retained<AnyObject>), MediaError> {
    // SAFETY: `+assetReaderWithAsset:error:` returns a reader or nil with an
    // NSError, which `error: _` turns into a Result.
    let reader: Result<Retained<AnyObject>, Retained<NSError>> =
        unsafe { msg_send![class(c"AVAssetReader")?, assetReaderWithAsset: &**asset, error: _] };
    let reader = Reader(reader.map_err(|e| MediaError::Decode {
        what: "video",
        detail: format!("AVAssetReader: {}", e.localizedDescription()),
    })?);
    // SAFETY: `+assetReaderTrackOutputWithTrack:outputSettings:` takes a track
    // and a nullable settings dictionary.
    let output: Retained<AnyObject> = unsafe {
        msg_send![
            class(c"AVAssetReaderTrackOutput")?,
            assetReaderTrackOutputWithTrack: &**track,
            outputSettings: settings
        ]
    };
    // SAFETY: both are live; `canAddOutput:` is checked before `addOutput:`,
    // which raises for an output that cannot be added.
    let can: bool = unsafe { msg_send![&*reader.0, canAddOutput: &*output] };
    if !can {
        return Err(MediaError::Decode {
            what: "video",
            detail: "AVAssetReader cannot read this track with these settings".into(),
        });
    }
    // SAFETY: as above.
    let () = unsafe { msg_send![&*reader.0, addOutput: &*output] };
    Ok((reader, output))
}

fn start_reading(reader: &Reader) -> Result<(), MediaError> {
    // SAFETY: `reader` is a live AVAssetReader that has not started reading.
    let started: bool = unsafe { msg_send![&*reader.0, startReading] };
    if started {
        Ok(())
    } else {
        Err(MediaError::Decode {
            what: "video",
            detail: format!("AVAssetReader would not start: {}", reader.error_text()),
        })
    }
}

/// `-[AVAssetReaderOutput copyNextSampleBuffer]`: the next sample buffer (+1),
/// or `None` at the end or on failure (the reader's status says which).
fn next_sample(output: &Retained<AnyObject>) -> Option<CfRef> {
    // SAFETY: `output` is a live AVAssetReaderTrackOutput on a started reader;
    // the method returns a CMSampleBufferRef the caller owns (CF_RETURNS_RETAINED)
    // or NULL, so it is taken as an owned CF reference.
    unsafe {
        let sb: *const OpaqueCmSampleBuffer = msg_send![&**output, copyNextSampleBuffer];
        CfRef::take(sb.cast())
    }
}

/// `{kCVPixelBufferPixelFormatTypeKey: @(format)}`.
fn format_settings(format: u32) -> Result<Retained<AnyObject>, MediaError> {
    // SAFETY: `+numberWithUnsignedInt:` and `+dictionaryWithObject:forKey:` are
    // plain Foundation constructors; the key is a CFString constant, which is
    // an NSString.
    unsafe {
        let number: Retained<AnyObject> = msg_send![class(c"NSNumber")?, numberWithUnsignedInt: format];
        let dict: Retained<AnyObject> = msg_send![
            class(c"NSDictionary")?,
            dictionaryWithObject: &*number,
            forKey: &*kCVPixelBufferPixelFormatTypeKey
        ];
        Ok(dict)
    }
}

/// `{kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)}`.
fn bgra_settings() -> Result<Retained<AnyObject>, MediaError> {
    format_settings(PIXEL_FORMAT_32BGRA)
}

/// What the first (undecoded) pass read: the media-time presentation time and
/// duration of every frame, in ticks of one timescale, in presentation order.
struct FirstPass {
    timescale: i32,
    pts: Vec<i64>,
    durations: Vec<i64>,
}

/// Pass one: read the track's compressed samples (no decoding) and return each
/// frame's presentation time and duration in presentation order.
fn count_frames(asset: &Retained<AnyObject>, track: &Retained<AnyObject>) -> Result<FirstPass, MediaError> {
    let (reader, output) = new_reader(asset, track, None)?;
    start_reading(&reader)?;
    let mut times: Vec<i64> = Vec::new();
    let mut durations: Vec<i64> = Vec::new();
    let mut timescale = 0i32;
    loop {
        let stop = autoreleasepool(|_| -> Result<bool, MediaError> {
            let Some(sample) = next_sample(&output) else {
                return Ok(true);
            };
            // SAFETY: `sample` is a live CMSampleBuffer.
            let n = unsafe { CMSampleBufferGetNumSamples(sample.as_ptr()) };
            if n == 0 {
                return Ok(false); // a marker-only buffer
            }
            if n != 1 {
                return Err(MediaError::Unsupported {
                    what: "video",
                    detail: format!("a sample buffer holds {n} samples; one frame per sample is expected"),
                });
            }
            // SAFETY: as above.
            let pts = unsafe { CMSampleBufferGetPresentationTimeStamp(sample.as_ptr()) };
            if pts.flags & CMTIME_VALID == 0 {
                return Err(MediaError::Decode {
                    what: "video",
                    detail: format!("frame {} has no valid presentation time", times.len()),
                });
            }
            // SAFETY: as above; the duration is returned by value.
            let duration = unsafe { CMSampleBufferGetDuration(sample.as_ptr()) };
            if duration.flags & CMTIME_VALID == 0 || duration.value < 0 {
                return Err(MediaError::Decode {
                    what: "video",
                    detail: format!("frame {} has no valid duration", times.len()),
                });
            }
            if pts.timescale <= 0
                || (timescale != 0 && pts.timescale != timescale)
                || duration.timescale != pts.timescale
            {
                return Err(MediaError::Unsupported {
                    what: "video",
                    detail: format!(
                        "frame {} counts time in 1/{} s where the first counted in 1/{timescale} s",
                        times.len(),
                        pts.timescale
                    ),
                });
            }
            timescale = pts.timescale;
            if times.len() >= MAX_VIDEO_FRAMES {
                return Err(MediaError::Limit {
                    what: "video",
                    detail: format!("more than {MAX_VIDEO_FRAMES} frames"),
                });
            }
            times.push(pts.value);
            durations.push(duration.value);
            Ok(false)
        })?;
        if stop {
            break;
        }
    }
    if reader.status() != READER_STATUS_COMPLETED {
        return Err(MediaError::Decode {
            what: "video",
            detail: format!(
                "reading the track stopped after {} frames: {}",
                times.len(),
                reader.error_text()
            ),
        });
    }
    if times.is_empty() {
        return Err(MediaError::Decode {
            what: "video",
            detail: "the video track has no frames".into(),
        });
    }
    // Presentation order: by time, equal times in the order read.
    let mut order: Vec<usize> = (0..times.len()).collect();
    order.sort_by_key(|&i| times[i]);
    Ok(FirstPass {
        timescale,
        pts: order.iter().map(|&i| times[i]).collect(),
        durations: order.iter().map(|&i| durations[i]).collect(),
    })
}

/// What the first decoded frame says about the track's decoded form.
struct FirstDecoded {
    /// Presentation time (track time, edits applied), seconds.
    seconds: f64,
    width: usize,
    height: usize,
}

/// Decode the first frame natively and report it; the reader is cancelled
/// afterwards.
fn first_decoded(
    asset: &Retained<AnyObject>,
    track: &Retained<AnyObject>,
    tags: StreamTags,
    layout: &Result<Layout, String>,
) -> Result<FirstDecoded, MediaError> {
    // The planes of an 8-bit 4:2:0 stream in the range it is tagged with (an
    // empty settings dictionary is refused by AVFoundation with an exception);
    // anything else as BGRA. Only the frame's time and size are used here.
    let eight_bit_420 = matches!(
        layout,
        Ok(Layout {
            chroma: Chroma::C420,
            depth: 8
        })
    );
    let settings = if eight_bit_420 {
        format_settings(if tags.full_range == Some(true) {
            PIXEL_FORMAT_420F
        } else {
            PIXEL_FORMAT_420V
        })?
    } else {
        bgra_settings()?
    };
    let (reader, output) = new_reader(asset, track, Some(&settings))?;
    start_reading(&reader)?;
    loop {
        let Some(sample) = next_sample(&output) else {
            return Err(MediaError::Decode {
                what: "video",
                detail: format!("no frame could be decoded: {}", reader.error_text()),
            });
        };
        // SAFETY: `sample` is a live CMSampleBuffer.
        let image = unsafe { CMSampleBufferGetImageBuffer(sample.as_ptr()) };
        if image.is_null() {
            continue;
        }
        // SAFETY: as above; the time is returned by value, and `image` is the
        // sample's live CVPixelBuffer.
        let (seconds, width, height) = unsafe {
            (
                CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample.as_ptr())),
                CVPixelBufferGetWidth(image),
                CVPixelBufferGetHeight(image),
            )
        };
        return if seconds.is_finite() {
            Ok(FirstDecoded { seconds, width, height })
        } else {
            Err(MediaError::Decode {
                what: "video",
                detail: "the first decoded frame has no presentation time".into(),
            })
        };
    }
}

/// A read-only lock on a pixel buffer's base addresses, unlocked when dropped.
struct PixelLock(CfTypeRef);

impl PixelLock {
    fn new(image: CfTypeRef) -> Result<PixelLock, MediaError> {
        // SAFETY: `image` is a live CVPixelBuffer (owned by the sample buffer the
        // caller holds); locking read-only makes the base addresses valid until
        // the matching unlock in `Drop`.
        let locked = unsafe { CVPixelBufferLockBaseAddress(image, PIXEL_BUFFER_LOCK_READ_ONLY) };
        if locked != 0 {
            return Err(MediaError::Decode {
                what: "video",
                detail: format!("CVPixelBufferLockBaseAddress failed ({locked})"),
            });
        }
        Ok(PixelLock(image))
    }
}

impl Drop for PixelLock {
    fn drop(&mut self) {
        // SAFETY: unlock the buffer locked in `new`, with the same flags.
        unsafe { CVPixelBufferUnlockBaseAddress(self.0, PIXEL_BUFFER_LOCK_READ_ONLY) };
    }
}

/// The size and pixel format of a pixel buffer, refused when empty or over the
/// pixel bound.
fn buffer_shape(image: CfTypeRef) -> Result<(u32, usize, usize), MediaError> {
    // SAFETY: `image` is a live CVPixelBuffer; the getters only read its description.
    let (format, w, h) = unsafe {
        (
            CVPixelBufferGetPixelFormatType(image),
            CVPixelBufferGetWidth(image),
            CVPixelBufferGetHeight(image),
        )
    };
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|p| p > MAX_IMAGE_PIXELS) {
        return Err(MediaError::Limit {
            what: "video",
            detail: format!("a {h}x{w} frame is outside 1..={MAX_IMAGE_PIXELS} pixels"),
        });
    }
    Ok((format, w, h))
}

/// Refuse a frame whose size (after rotation) is not the first frame's.
fn check_size(
    (w, h): (usize, usize),
    orientation: u32,
    rotation: Rotation,
    expect: Option<(usize, usize)>,
) -> Result<(), MediaError> {
    if let Some((ew, eh)) = expect {
        let (rw, rh) = if rotation == Rotation::Apply {
            orient::oriented_size(orientation, w, h)
        } else {
            (w, h)
        };
        if (rw, rh) != (ew, eh) {
            return Err(MediaError::Decode {
                what: "video",
                detail: format!("a frame is {rh}x{rw} where the first was {eh}x{ew}"),
            });
        }
    }
    Ok(())
}

/// An RGB frame and its width and height, or why it could not be made.
type Converted = Result<(Vec<u8>, usize, usize), MediaError>;

/// One decoded BGRA pixel buffer (AVFoundation's own conversion) as an RGB
/// frame, rotated per `orientation` when `rotation` is [`Rotation::Apply`];
/// `expect` is the size of the frames before it, which every frame must share.
fn convert_frame(image: CfTypeRef, orientation: u32, rotation: Rotation, expect: Option<(usize, usize)>) -> Converted {
    let (format, w, h) = buffer_shape(image)?;
    check_size((w, h), orientation, rotation, expect)?;
    let rgb = bgra_to_rgb(image, format, w, h)?;
    if rotation == Rotation::Apply && orientation != 1 {
        return Ok(orient::apply(orientation, rgb, w, h, 3));
    }
    Ok((rgb, w, h))
}

/// A plane's samples moved by EXIF orientation `o`: `units` samples make one
/// pixel (1 for luma, 2 for an interleaved Cb, Cr pair); returns the new plane
/// and its width and height in pixels.
fn orient_plane(o: u32, plane: &Samples, width: usize, height: usize, units: usize) -> (Samples, usize, usize) {
    match plane {
        Samples::U8(v) => {
            let (out, w, h) = orient::apply(o, v.clone(), width, height, units);
            (Samples::U8(out), w, h)
        }
        Samples::U16(v) => {
            let bytes: Vec<u8> = v.iter().flat_map(|s| s.to_le_bytes()).collect();
            let (out, w, h) = orient::apply(o, bytes, width, height, 2 * units);
            let words = out.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            (Samples::U16(words), w, h)
        }
    }
}

/// Convert planes of `width x height` (tightly packed, as [`NativeFrame`]).
fn convert_planes(
    width: usize,
    height: usize,
    layout: Layout,
    y: &Samples,
    uv: &Samples,
    params: &Params,
) -> Result<Vec<u8>, MediaError> {
    // Interleaved Cb, Cr pairs: one per sample for 4:4:4, one per two for the others.
    let uv_stride = if layout.chroma == Chroma::C444 {
        2 * width
    } else {
        width
    };
    let convert_err = |e: yuv::YuvError| MediaError::Decode {
        what: "video",
        detail: e.to_string(),
    };
    match (y, uv) {
        (Samples::U8(y), Samples::U8(uv)) => yuv::to_rgb24(
            &Picture {
                width,
                height,
                layout,
                y,
                y_stride: width,
                uv,
                uv_stride,
            },
            params,
        ),
        (Samples::U16(y), Samples::U16(uv)) => yuv::to_rgb24(
            &Picture {
                width,
                height,
                layout,
                y,
                y_stride: width,
                uv,
                uv_stride,
            },
            params,
        ),
        _ => {
            return Err(MediaError::Decode {
                what: "video",
                detail: "a frame's planes hold different sample types".into(),
            })
        }
    }
    .map_err(convert_err)
}

/// Decoded planes as an RGB frame by libswscale's arithmetic (`params` is
/// matrix, range, flavor, chroma filter and threads), rotated per `orientation`
/// when `rotation` is [`Rotation::Apply`] in the order the `ffmpeg` CLI's
/// autorotate has, which the bytes depend on for some layouts: for 4:2:0
/// (whose planes it can turn) the planes are turned first and the converted
/// picture has the turned size (the converter, its width-dependent choice and,
/// at 10 bits, the vertical chroma filter, act on the upright picture); for 4:2:2
/// and 4:4:4 the CLI converts first and turns the RGB. Measured on all three at
/// 8 and 10 bits; whether torchcodec orders the same is not known.
fn convert_native(
    frame: &NativeFrame,
    params: &Params,
    orientation: u32,
    rotation: Rotation,
    expect: Option<(usize, usize)>,
) -> Converted {
    let (w, h) = (frame.width, frame.height);
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|p| p > MAX_IMAGE_PIXELS) {
        return Err(MediaError::Limit {
            what: "video",
            detail: format!("a {h}x{w} frame is outside 1..={MAX_IMAGE_PIXELS} pixels"),
        });
    }
    check_size((w, h), orientation, rotation, expect)?;
    let turn = rotation == Rotation::Apply && orientation != 1;
    if turn && frame.layout.chroma == Chroma::C420 {
        let (y, tw, th) = orient_plane(orientation, &frame.y, w, h, 1);
        let (uv, ..) = orient_plane(orientation, &frame.uv, w / 2, h / 2, 2);
        let rgb = convert_planes(tw, th, frame.layout, &y, &uv, params)?;
        return Ok((rgb, tw, th));
    }
    let rgb = convert_planes(w, h, frame.layout, &frame.y, &frame.uv, params)?;
    if turn {
        return Ok(orient::apply(orientation, rgb, w, h, 3));
    }
    Ok((rgb, w, h))
}

/// A locked BGRA pixel buffer as tightly packed RGB.
fn bgra_to_rgb(image: CfTypeRef, format: u32, w: usize, h: usize) -> Result<Vec<u8>, MediaError> {
    if format != PIXEL_FORMAT_32BGRA {
        return Err(MediaError::Decode {
            what: "video",
            detail: format!("AVFoundation returned pixel format {format:#x}, not BGRA"),
        });
    }
    let lock = PixelLock::new(image)?;
    // SAFETY: the buffer is locked (`lock`), so the base address and stride are valid.
    let (base, stride) = unsafe { (CVPixelBufferGetBaseAddress(image), CVPixelBufferGetBytesPerRow(image)) };
    if base.is_null() || stride < w * 4 {
        return Err(MediaError::Decode {
            what: "video",
            detail: format!("the pixel buffer has no readable rows (stride {stride} for {w} pixels)"),
        });
    }
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        // SAFETY: the locked buffer holds `h` rows of `stride >= 4 * w` bytes
        // starting at `base`; this reads the first `4 * w` of row `y`.
        let row = unsafe { std::slice::from_raw_parts(base.add(y * stride), w * 4) };
        let dst = &mut rgb[y * w * 3..(y + 1) * w * 3];
        for (px, out) in row.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
            out[0] = px[2];
            out[1] = px[1];
            out[2] = px[0];
        }
    }
    drop(lock);
    Ok(rgb)
}
#[cfg(test)]
mod tests {
    use objc2::encode::Encode;
    use objc2::runtime::{AnyClass, Sel};
    use objc2::sel;

    use super::{CgAffineTransform, OpaqueCmSampleBuffer};

    /// What the Objective-C runtime says an instance method of `class` returns.
    fn runtime_return(class: &std::ffi::CStr, sel: Sel) -> String {
        let cls = AnyClass::get(class).unwrap_or_else(|| panic!("{class:?} is not loaded"));
        let method = cls
            .instance_method(sel)
            .unwrap_or_else(|| panic!("{class:?} has no method {sel:?}"));
        method.return_type().to_str().expect("an ASCII encoding").to_owned()
    }

    /// objc2 compares a `msg_send!` return type with the method's encoding only
    /// in debug builds, so a wrong type passes every release test and panics in
    /// the first debug run (`^v` for `^{opaqueCMSampleBuffer=}` did). This holds
    /// the Rust encodings used for AVFoundation returns to the runtime's, in
    /// every profile.
    #[test]
    fn the_encodings_used_for_avfoundation_returns_are_the_runtimes() {
        assert_eq!(
            runtime_return(c"AVAssetReaderOutput", sel!(copyNextSampleBuffer)),
            <*const OpaqueCmSampleBuffer as Encode>::ENCODING.to_string(),
        );
        assert_eq!(
            runtime_return(c"AVAssetTrack", sel!(preferredTransform)),
            CgAffineTransform::ENCODING.to_string(),
        );
        assert_eq!(
            runtime_return(c"AVAssetTrack", sel!(nominalFrameRate)),
            f32::ENCODING.to_string()
        );
    }
}
