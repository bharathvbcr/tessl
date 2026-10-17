//! Video decoding with AVFoundation (`AVAssetReader`).
//!
//! H.264, HEVC and whatever else AVFoundation decodes in MP4, MOV and M4V
//! become decoded 8-bit RGB frames, sampled the way
//! [`crate::embedgemma2_pre::sample_frame_indices`] samples them, with the
//! source frame rate and each frame's presentation time.
//!
//! # How the frames are chosen
//!
//! The model's video processor needs the frame count and the source rate first
//! (`int(duration * fps)` frames at `int(i * source_fps / fps)`), then a handful
//! of frames. [`VideoSource::open`] therefore reads the track twice:
//!
//! 1. once **without decoding** (an `AVAssetReaderTrackOutput` with no output
//!    settings returns the compressed samples): the exact frame count, each
//!    frame's presentation time in presentation order, and the track's
//!    `nominalFrameRate` and `preferredTransform`;
//! 2. once **decoding** to `kCVPixelFormatType_32BGRA`, converting to RGB only
//!    the frames asked for.
//!
//! Decoding still runs through every frame (a P-frame cannot be skipped), but
//! only the sampled frames are converted and kept, so memory is
//! `sampled * width * height * 3`, not the whole video.
//!
//! # Colour and rotation
//!
//! The YCbCr to RGB conversion is AVFoundation's, using the matrix and range
//! the stream is tagged with. The frames come out as stored; the track's
//! `preferredTransform` is applied by this module as one of the eight EXIF
//! orientations ([`Rotation::Apply`], the default, as a player and OpenCV's
//! `cv2.VideoCapture` do). Which one torchcodec's `VideoDecoder` does by default
//! could not be checked here (it is not installed), so [`Rotation::Ignore`]
//! returns the frames as stored.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use block2::RcBlock;
use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::msg_send;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyClass, AnyObject};
use objc2_foundation::{NSError, NSString, NSURL};

use super::cf::{CfRef, CfTypeRef};
use super::{containers, orient, MediaError};
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
struct CmTime {
    value: i64,
    timescale: i32,
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
    fn CMSampleBufferGetNumSamples(sbuf: CfTypeRef) -> isize;
    fn CMSampleBufferGetPresentationTimeStamp(sbuf: CfTypeRef) -> CmTime;
    fn CMSampleBufferGetImageBuffer(sbuf: CfTypeRef) -> CfTypeRef;
    fn CMTimeGetSeconds(time: CmTime) -> f64;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: *const AnyObject;

    fn CVPixelBufferLockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: CfTypeRef) -> *const u8;
    fn CVPixelBufferGetBytesPerRow(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetWidth(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetHeight(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetPixelFormatType(pb: CfTypeRef) -> u32;
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

/// Options for opening and decoding a video.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoOptions {
    pub rotation: Rotation,
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
            max_frames: 4096,
            max_bytes: 1 << 32,
            load_timeout: Duration::from_secs(60),
        }
    }
}

/// What the first (undecoded) pass found.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    /// The exact number of frames in the video track.
    pub total_frames: usize,
    /// The source frame rate: the track's `nominalFrameRate`, or the average
    /// rate over the presentation times when the track does not give one.
    pub fps: f64,
    /// Each frame's presentation time in seconds, in presentation order.
    pub timestamps: Vec<f64>,
    /// The EXIF orientation (1 to 8) the track's `preferredTransform` stands
    /// for; 1 when it is the identity.
    pub orientation: u32,
}

/// Decoded frames and where they came from.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedVideo {
    /// Width of every frame (after rotation, if applied).
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    /// Frames in the whole video.
    pub total_frames: usize,
    /// Source index (presentation order) of each returned frame.
    pub indices: Vec<usize>,
    /// Presentation time of each returned frame, seconds.
    pub timestamps: Vec<f64>,
    /// One `height * width * 3` RGB buffer per returned frame.
    pub frames: Vec<Vec<u8>>,
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
        const WHAT: &str = "video";
        if indices.is_empty() {
            return Err(MediaError::Invalid {
                what: WHAT,
                detail: "no frame indices asked for".into(),
            });
        }
        if indices.len() > self.options.max_frames {
            return Err(MediaError::Limit {
                what: WHAT,
                detail: format!("{} frames exceeds {}", indices.len(), self.options.max_frames),
            });
        }
        if let Some(&bad) = indices.iter().find(|&&i| i >= self.info.total_frames) {
            return Err(MediaError::Invalid {
                what: WHAT,
                detail: format!("frame {bad} asked for, the video has {}", self.info.total_frames),
            });
        }
        let mut wanted: Vec<usize> = indices.to_vec();
        wanted.sort_unstable();
        wanted.dedup();
        autoreleasepool(|_| self.decode_wanted(&wanted, indices))
    }

    /// Decode the frames the model's video processor samples
    /// ([`sample_frame_indices`] over the exact frame count and source rate);
    /// with `cfg.do_sample_frames` false, every frame (up to `max_frames`).
    ///
    /// # Errors
    /// As [`VideoSource::decode_frames`], and the sampler's own errors.
    pub fn decode_sampled(&self, cfg: &VideoConfig) -> Result<DecodedVideo, MediaError> {
        let indices = if cfg.do_sample_frames {
            sample_frame_indices(cfg, self.info.total_frames, self.info.fps).map_err(MediaError::Preprocess)?
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

    /// Pass two: decode and convert the sorted, unique `wanted` frames, then lay
    /// them out in the order of `asked`.
    fn decode_wanted(&self, wanted: &[usize], asked: &[usize]) -> Result<DecodedVideo, MediaError> {
        const WHAT: &str = "video";
        let settings = bgra_settings()?;
        let (reader, output) = new_reader(&self.asset, &self.track, Some(&settings))?;
        start_reading(&reader)?;
        let mut kept: Vec<Vec<u8>> = Vec::with_capacity(wanted.len());
        let mut dims: Option<(usize, usize)> = None;
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
                    let (rgb, w, h) = convert_frame(image, self.info.orientation, self.options.rotation, dims)?;
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
                    kept.push(rgb);
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
        })
    }
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
        let orientation = match orient::orientation_of_transform(transform.a, transform.b, transform.c, transform.d) {
            Some(o) => o,
            None if options.rotation == Rotation::Ignore => 1,
            None => {
                return Err(MediaError::Unsupported {
                    what: "video",
                    detail: format!(
                        "the track's transform [{} {} {} {}] is not a right-angle rotation or flip",
                        transform.a, transform.b, transform.c, transform.d
                    ),
                })
            }
        };
        let mut timestamps = count_frames(&asset, &track)?;
        // Compressed samples carry media time; decoded frames carry track time,
        // with the edit list applied (an H.264 stream with B-frames starts its
        // media at +2 frames and the edit list shifts it back to 0). Report
        // track time, which is what FFmpeg and OpenCV report, and which
        // `decode_wanted` checks against every decoded frame.
        let shift = timestamps[0] - first_decoded_time(&asset, &track)?;
        for t in &mut timestamps {
            *t -= shift;
        }
        let total_frames = timestamps.len();
        let fps = if track_frame_rate.is_finite() && track_frame_rate > 0.0 {
            f64::from(track_frame_rate)
        } else if total_frames > 1 && timestamps[total_frames - 1] > timestamps[0] {
            (total_frames - 1) as f64 / (timestamps[total_frames - 1] - timestamps[0])
        } else {
            return Err(MediaError::Decode {
                what: "video",
                detail: "the track has no frame rate and too few frames to derive one".into(),
            });
        };
        Ok(VideoSource {
            asset,
            track,
            info: VideoInfo {
                total_frames,
                fps,
                timestamps,
                orientation,
            },
            options: *options,
            _temp: temp,
        })
    })
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

/// `{kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)}`.
fn bgra_settings() -> Result<Retained<AnyObject>, MediaError> {
    // SAFETY: `+numberWithUnsignedInt:` and `+dictionaryWithObject:forKey:` are
    // plain Foundation constructors; the key is a CFString constant, which is
    // an NSString.
    unsafe {
        let number: Retained<AnyObject> = msg_send![class(c"NSNumber")?, numberWithUnsignedInt: PIXEL_FORMAT_32BGRA];
        let dict: Retained<AnyObject> = msg_send![
            class(c"NSDictionary")?,
            dictionaryWithObject: &*number,
            forKey: &*kCVPixelBufferPixelFormatTypeKey
        ];
        Ok(dict)
    }
}

/// Pass one: read the track's compressed samples (no decoding) and return each
/// frame's presentation time in presentation order.
fn count_frames(asset: &Retained<AnyObject>, track: &Retained<AnyObject>) -> Result<Vec<f64>, MediaError> {
    let (reader, output) = new_reader(asset, track, None)?;
    start_reading(&reader)?;
    let mut times: Vec<f64> = Vec::new();
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
            // SAFETY: `pts` is a valid CMTime by value.
            let seconds = unsafe { CMTimeGetSeconds(pts) };
            if !seconds.is_finite() {
                return Err(MediaError::Decode {
                    what: "video",
                    detail: format!("frame {} has a non-finite presentation time", times.len()),
                });
            }
            if times.len() >= MAX_VIDEO_FRAMES {
                return Err(MediaError::Limit {
                    what: "video",
                    detail: format!("more than {MAX_VIDEO_FRAMES} frames"),
                });
            }
            times.push(seconds);
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
    times.sort_by(f64::total_cmp);
    Ok(times)
}

/// The presentation time (track time, edits applied) of the first decoded
/// frame; the reader is cancelled afterwards.
fn first_decoded_time(asset: &Retained<AnyObject>, track: &Retained<AnyObject>) -> Result<f64, MediaError> {
    let settings = bgra_settings()?;
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
        if unsafe { CMSampleBufferGetImageBuffer(sample.as_ptr()) }.is_null() {
            continue;
        }
        // SAFETY: as above; the time is returned by value.
        let seconds = unsafe { CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample.as_ptr())) };
        return if seconds.is_finite() {
            Ok(seconds)
        } else {
            Err(MediaError::Decode {
                what: "video",
                detail: "the first decoded frame has no presentation time".into(),
            })
        };
    }
}

/// One decoded BGRA pixel buffer as an RGB frame, rotated per `orientation`
/// when `rotation` is [`Rotation::Apply`]; `expect` is the size of the frames
/// before it, which every frame must share.
fn convert_frame(
    image: CfTypeRef,
    orientation: u32,
    rotation: Rotation,
    expect: Option<(usize, usize)>,
) -> Result<(Vec<u8>, usize, usize), MediaError> {
    // SAFETY: `image` is a live CVPixelBuffer (owned by the sample buffer the
    // caller holds); the getters only read its description.
    let (format, w, h) = unsafe {
        (
            CVPixelBufferGetPixelFormatType(image),
            CVPixelBufferGetWidth(image),
            CVPixelBufferGetHeight(image),
        )
    };
    if format != PIXEL_FORMAT_32BGRA {
        return Err(MediaError::Decode {
            what: "video",
            detail: format!("AVFoundation returned pixel format {format:#x}, not BGRA"),
        });
    }
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|p| p > MAX_IMAGE_PIXELS) {
        return Err(MediaError::Limit {
            what: "video",
            detail: format!("a {h}x{w} frame is outside 1..={MAX_IMAGE_PIXELS} pixels"),
        });
    }
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
    // SAFETY: locking read-only makes the base address valid until the matching
    // unlock below, on every path out of this block.
    let locked = unsafe { CVPixelBufferLockBaseAddress(image, PIXEL_BUFFER_LOCK_READ_ONLY) };
    if locked != 0 {
        return Err(MediaError::Decode {
            what: "video",
            detail: format!("CVPixelBufferLockBaseAddress failed ({locked})"),
        });
    }
    // SAFETY: the buffer is locked, so the base address and stride are valid.
    let (base, stride) = unsafe { (CVPixelBufferGetBaseAddress(image), CVPixelBufferGetBytesPerRow(image)) };
    let mut rgb = Vec::new();
    if !base.is_null() && stride >= w * 4 {
        rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            // SAFETY: the locked buffer holds `h` rows of `stride >= 4 * w`
            // bytes starting at `base`; this reads the first `4 * w` of row `y`.
            let row = unsafe { std::slice::from_raw_parts(base.add(y * stride), w * 4) };
            let dst = &mut rgb[y * w * 3..(y + 1) * w * 3];
            for (px, out) in row.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
                out[0] = px[2];
                out[1] = px[1];
                out[2] = px[0];
            }
        }
    }
    // SAFETY: unlock the buffer locked above with the same flags.
    unsafe { CVPixelBufferUnlockBaseAddress(image, PIXEL_BUFFER_LOCK_READ_ONLY) };
    if rgb.is_empty() {
        return Err(MediaError::Decode {
            what: "video",
            detail: format!("the pixel buffer has no readable rows (stride {stride} for {w} pixels)"),
        });
    }
    if rotation == Rotation::Apply && orientation != 1 {
        return Ok(orient::apply(orientation, rgb, w, h, 3));
    }
    Ok((rgb, w, h))
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
