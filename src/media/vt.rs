//! Decoding a track's compressed samples with VideoToolbox, to the decoder's own
//! 8-bit 4:2:0 planes.
//!
//! # Why not `AVAssetReader` for this
//!
//! `AVAssetReaderTrackOutput` can vend decoded `420v`/`420f` pixel buffers, and
//! their luma is the decoder's. Their chroma is not: against FFmpeg's decode of
//! the same file (and against the source of a lossless stream), 8 to 15 percent
//! of the chroma samples differ, by up to 70 levels at colour edges and by one or
//! two in gradients, for H.264 and HEVC, with deblocking off, on intra-only
//! streams, and with `AVVideoColorPropertiesKey` set. The reader post-processes
//! the decoder's output (it is not siting: no whole-sample shift lines the planes
//! up). The same decoder driven directly returns FFmpeg's bytes: FFmpeg's own
//! `-hwaccel videotoolbox` output is byte-identical to its software decode, and so
//! is this module's (`tests/media_video_exact.rs`).
//!
//! So `AVAssetReader` supplies the *compressed* samples (settings `nil`: decode
//! order, with the track's format description), and a `VTDecompressionSession`
//! decodes them. Every sample is decoded, because later frames depend on earlier
//! ones; only the frames asked for are copied out. The session emits frames in
//! decode order, each with the presentation time of its sample, which names its
//! index in presentation order.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;

use super::cf::{CFDictionaryGetValue, CFGetTypeID, CFNumberGetTypeID, CFNumberGetValue, CfRef, CfTypeRef};
use super::video::{CmTime, Nv12Frame};
use super::MediaError;

/// `kCFNumberFloat64Type`.
const CF_NUMBER_FLOAT64_TYPE: isize = 6;
/// `kCVPixelBufferLock_ReadOnly`.
const LOCK_READ_ONLY: u64 = 1;
/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange` ('420v') and `FullRange` ('420f').
pub(super) const FORMAT_420V: u32 = u32::from_be_bytes(*b"420v");
pub(super) const FORMAT_420F: u32 = u32::from_be_bytes(*b"420f");
/// `kVTDecodeInfo_FrameDropped`.
const INFO_FRAME_DROPPED: u32 = 1 << 1;

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    fn VTDecompressionSessionCreate(
        allocator: CfTypeRef,
        format_description: CfTypeRef,
        decoder_specification: CfTypeRef,
        destination_image_buffer_attributes: CfTypeRef,
        output_callback: *const c_void,
        session_out: *mut CfTypeRef,
    ) -> i32;
    fn VTDecompressionSessionDecodeFrameWithOutputHandler(
        session: CfTypeRef,
        sample_buffer: CfTypeRef,
        decode_flags: u32,
        info_flags_out: *mut u32,
        output_handler: *const c_void,
    ) -> i32;
    fn VTDecompressionSessionWaitForAsynchronousFrames(session: CfTypeRef) -> i32;
    fn VTDecompressionSessionInvalidate(session: CfTypeRef);
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    fn CMSampleBufferGetFormatDescription(sbuf: CfTypeRef) -> CfTypeRef;
    fn CMSampleBufferGetPresentationTimeStamp(sbuf: CfTypeRef) -> CmTime;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: *const AnyObject;
    static kCVImageBufferCleanApertureKey: CfTypeRef;
    static kCVImageBufferCleanApertureWidthKey: CfTypeRef;
    static kCVImageBufferCleanApertureHeightKey: CfTypeRef;
    static kCVImageBufferCleanApertureHorizontalOffsetKey: CfTypeRef;
    static kCVImageBufferCleanApertureVerticalOffsetKey: CfTypeRef;

    fn CVPixelBufferLockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferGetPixelFormatType(pb: CfTypeRef) -> u32;
    fn CVPixelBufferGetWidth(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetHeight(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetPlaneCount(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetWidthOfPlane(pb: CfTypeRef, plane: usize) -> usize;
    fn CVPixelBufferGetHeightOfPlane(pb: CfTypeRef, plane: usize) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(pb: CfTypeRef, plane: usize) -> *const u8;
    fn CVPixelBufferGetBytesPerRowOfPlane(pb: CfTypeRef, plane: usize) -> usize;
    /// `CVBufferCopyAttachment` (macOS 12): the attachment (+1) or NULL.
    fn CVBufferCopyAttachment(buffer: CfTypeRef, key: CfTypeRef, mode: *mut u32) -> CfTypeRef;
}

/// What the output handler collects. Its fields are touched by VideoToolbox's
/// thread while frames decode and by the caller afterwards, so they sit behind a
/// mutex.
struct Shared {
    /// Presentation times (media ticks) of the frames in presentation order.
    pts: Vec<i64>,
    /// Which frames to copy out.
    wanted: Vec<bool>,
    frames: Vec<Option<Nv12Frame>>,
    outputs: usize,
    /// The first failure, as text.
    error: Option<String>,
}

/// A decoding session over one track. Invalidated and released when dropped, on
/// every path.
pub(super) struct Decoder {
    session: CfRef,
    description: CfTypeRef,
    shared: Arc<Mutex<Shared>>,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: `session` is a live VTDecompressionSession this decoder owns;
        // invalidating it before the release (by `CfRef`'s drop, after this runs)
        // is the documented teardown and stops further handler calls.
        unsafe { VTDecompressionSessionInvalidate(self.session.as_ptr()) };
    }
}

fn framework(api: &'static str, detail: impl Into<String>) -> MediaError {
    MediaError::Framework {
        api,
        detail: detail.into(),
    }
}

impl Decoder {
    /// A session for the track `first` (a `CMSampleBufferRef`) belongs to,
    /// decoding to pixel format `format` ([`FORMAT_420V`] or [`FORMAT_420F`]).
    /// `pts` are the presentation times of the track's frames in presentation
    /// order, `wanted` which of them to keep.
    pub(super) fn new(first: CfTypeRef, format: u32, pts: Vec<i64>, wanted: Vec<bool>) -> Result<Decoder, MediaError> {
        // SAFETY: `first` is a live CMSampleBuffer; the format description is
        // borrowed from it and kept alive by the caller's samples.
        let description = unsafe { CMSampleBufferGetFormatDescription(first) };
        if description.is_null() {
            return Err(framework(
                "CMSampleBufferGetFormatDescription",
                "the sample has no format description",
            ));
        }
        let attributes = format_attributes(format)?;
        let mut session: CfTypeRef = std::ptr::null();
        // SAFETY: the description is live; a NULL allocator, specification and
        // callback record are the defaults (the callback must be NULL because
        // frames are decoded with output handlers); `session` receives a +1
        // session or stays NULL on failure.
        let status = unsafe {
            VTDecompressionSessionCreate(
                std::ptr::null(),
                description,
                std::ptr::null(),
                (&*attributes as *const AnyObject).cast(),
                std::ptr::null(),
                &mut session,
            )
        };
        // SAFETY: on success `session` is a +1 reference we now own.
        let session = unsafe { CfRef::take(session) }
            .filter(|_| status == 0)
            .ok_or_else(|| framework("VTDecompressionSessionCreate", format!("OSStatus {status}")))?;
        let frames = (0..pts.len()).map(|_| None).collect();
        Ok(Decoder {
            session,
            description,
            shared: Arc::new(Mutex::new(Shared {
                pts,
                wanted,
                frames,
                outputs: 0,
                error: None,
            })),
        })
    }

    /// Decode one compressed sample. Samples must come in decode order and share
    /// the first one's format description.
    pub(super) fn decode(&mut self, sample: CfTypeRef, format: u32) -> Result<(), MediaError> {
        // SAFETY: `sample` is a live CMSampleBuffer.
        let description = unsafe { CMSampleBufferGetFormatDescription(sample) };
        // SAFETY: both are live CMFormatDescriptions (the session's was borrowed
        // from a sample the caller still holds).
        if description.is_null()
            || (description != self.description && unsafe { super::cf::CFEqual(description, self.description) } == 0)
        {
            return Err(MediaError::Unsupported {
                what: "video",
                detail: "the track changes its format description mid-stream".into(),
            });
        }
        // Each sample has its own handler, so the frame a handler receives is
        // the one this sample codes: its index is found from the sample's time
        // here, not from the time VideoToolbox reports with the output.
        // SAFETY: `sample` is a live CMSampleBuffer; the time is returned by value.
        let sample_pts = unsafe { CMSampleBufferGetPresentationTimeStamp(sample) };
        let index = {
            let guard = match self.shared.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard
                .pts
                .binary_search(&sample_pts.value)
                .map_err(|_| MediaError::Decode {
                    what: "video",
                    detail: format!(
                        "a sample at time {}/{} is not one the first pass counted",
                        sample_pts.value, sample_pts.timescale
                    ),
                })?
        };
        let shared = Arc::clone(&self.shared);
        let handler = RcBlock::new(
            move |status: i32, info: u32, image: *const c_void, _pts: CmTime, _duration: CmTime| {
                let mut guard = match shared.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                let result = (|| -> Result<(), String> {
                    if status != 0 {
                        return Err(format!("OSStatus {status}"));
                    }
                    if info & INFO_FRAME_DROPPED != 0 || image.is_null() {
                        return Err("a frame was dropped".into());
                    }
                    guard.outputs += 1;
                    if guard.wanted[index] {
                        if guard.frames[index].is_some() {
                            return Err(format!("frame {index} came out twice"));
                        }
                        guard.frames[index] = Some(copy_planes(image, format)?);
                    }
                    Ok(())
                })();
                if let Err(e) = result {
                    guard.error.get_or_insert(e);
                }
            },
        );
        let mut info = 0u32;
        // SAFETY: the session and sample are live; the block lives until the
        // call returns and VideoToolbox copies it for later delivery; `info` is
        // a valid out-pointer.
        let status = unsafe {
            VTDecompressionSessionDecodeFrameWithOutputHandler(
                self.session.as_ptr(),
                sample,
                0,
                &mut info,
                (&*handler as *const _) as *const c_void,
            )
        };
        if status != 0 {
            return Err(MediaError::Decode {
                what: "video",
                detail: format!("VTDecompressionSessionDecodeFrame: OSStatus {status}"),
            });
        }
        self.failure()
    }

    fn failure(&self) -> Result<(), MediaError> {
        let guard = match self.shared.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match &guard.error {
            Some(e) => Err(MediaError::Decode {
                what: "video",
                detail: format!("VideoToolbox: {e}"),
            }),
            None => Ok(()),
        }
    }

    /// Wait for the last frames and return the kept ones, indexed by presentation order.
    pub(super) fn finish(self) -> Result<Vec<Option<Nv12Frame>>, MediaError> {
        // SAFETY: the session is live.
        let status = unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session.as_ptr()) };
        if status != 0 {
            return Err(MediaError::Decode {
                what: "video",
                detail: format!("VTDecompressionSessionWaitForAsynchronousFrames: OSStatus {status}"),
            });
        }
        self.failure()?;
        let mut guard = match self.shared.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if guard.outputs != guard.pts.len() {
            return Err(MediaError::Decode {
                what: "video",
                detail: format!(
                    "VideoToolbox returned {} frames for {} samples",
                    guard.outputs,
                    guard.pts.len()
                ),
            });
        }
        Ok(std::mem::take(&mut guard.frames))
    }
}

/// `{kCVPixelBufferPixelFormatTypeKey: @(format)}`.
fn format_attributes(format: u32) -> Result<Retained<AnyObject>, MediaError> {
    let class = |name: &'static std::ffi::CStr| {
        objc2::runtime::AnyClass::get(name)
            .ok_or_else(|| framework("objc_getClass", format!("{name:?} is not available")))
    };
    // SAFETY: `+numberWithUnsignedInt:` and `+dictionaryWithObject:forKey:` are
    // plain Foundation constructors; the key is a CFString constant, an NSString.
    unsafe {
        let number: Retained<AnyObject> = objc2::msg_send![class(c"NSNumber")?, numberWithUnsignedInt: format];
        let dict: Retained<AnyObject> = objc2::msg_send![
            class(c"NSDictionary")?,
            dictionaryWithObject: &*number,
            forKey: &*kCVPixelBufferPixelFormatTypeKey
        ];
        Ok(dict)
    }
}

/// A read-only lock on a pixel buffer, unlocked when dropped.
struct Lock(CfTypeRef);

impl Lock {
    fn new(image: CfTypeRef) -> Result<Lock, String> {
        // SAFETY: `image` is a live CVPixelBuffer for the handler's duration.
        let status = unsafe { CVPixelBufferLockBaseAddress(image, LOCK_READ_ONLY) };
        if status != 0 {
            return Err(format!("CVPixelBufferLockBaseAddress failed ({status})"));
        }
        Ok(Lock(image))
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        // SAFETY: unlocks the buffer locked in `new`, with the same flags.
        unsafe { CVPixelBufferUnlockBaseAddress(self.0, LOCK_READ_ONLY) };
    }
}

/// A `CFNumber` found in `dict` under `key`, as a double.
///
/// # Safety
/// `dict` must be a live `CFDictionary` and `key` a live CF string constant.
unsafe fn dict_f64(dict: CfTypeRef, key: CfTypeRef) -> Option<f64> {
    let value = CFDictionaryGetValue(dict, key);
    if value.is_null() || CFGetTypeID(value) != CFNumberGetTypeID() {
        return None;
    }
    let mut out = 0f64;
    // SAFETY: `value` is a CFNumber (checked) and `out` is an f64, the size
    // kCFNumberFloat64Type names.
    (CFNumberGetValue(value, CF_NUMBER_FLOAT64_TYPE, (&mut out as *mut f64).cast()) != 0).then_some(out)
}

/// The rectangle of a decoded buffer that is the picture: the clean aperture the
/// decoder attaches (H.264 and HEVC cropping), or the whole buffer without one.
fn picture_rect(image: CfTypeRef, coded: (usize, usize)) -> Result<(usize, usize, usize, usize), String> {
    // SAFETY: `image` is live; the attachment is +1 or NULL and is held by a `CfRef`.
    let aperture = unsafe {
        CfRef::take(CVBufferCopyAttachment(
            image,
            kCVImageBufferCleanApertureKey,
            std::ptr::null_mut(),
        ))
    };
    let Some(aperture) = aperture else {
        return Ok((0, 0, coded.0, coded.1));
    };
    // SAFETY: the aperture is a live CFDictionary and the keys are SDK constants.
    let (w, h, dx, dy) = unsafe {
        (
            dict_f64(aperture.as_ptr(), kCVImageBufferCleanApertureWidthKey),
            dict_f64(aperture.as_ptr(), kCVImageBufferCleanApertureHeightKey),
            dict_f64(aperture.as_ptr(), kCVImageBufferCleanApertureHorizontalOffsetKey),
            dict_f64(aperture.as_ptr(), kCVImageBufferCleanApertureVerticalOffsetKey),
        )
    };
    let (Some(w), Some(h), Some(dx), Some(dy)) = (w, h, dx, dy) else {
        return Err("the clean aperture is incomplete".into());
    };
    // The aperture's centre is offset from the buffer's centre by (dx, dy).
    let x0 = (coded.0 as f64 - w) / 2.0 + dx;
    let y0 = (coded.1 as f64 - h) / 2.0 + dy;
    let whole = |v: f64| v >= 0.0 && (v - v.round()).abs() < 1e-6;
    if !(whole(x0) && whole(y0) && whole(w) && whole(h)) || w < 1.0 || h < 1.0 {
        return Err(format!(
            "the clean aperture {w}x{h} at ({x0}, {y0}) is not on whole pixels"
        ));
    }
    let (x0, y0, w, h) = (
        x0.round() as usize,
        y0.round() as usize,
        w.round() as usize,
        h.round() as usize,
    );
    if x0 + w > coded.0 || y0 + h > coded.1 || x0 % 2 != 0 || y0 % 2 != 0 || w % 2 != 0 || h % 2 != 0 {
        return Err(format!(
            "the clean aperture {w}x{h} at ({x0}, {y0}) does not fit a {}x{} 4:2:0 buffer",
            coded.0, coded.1
        ));
    }
    Ok((x0, y0, w, h))
}

/// The picture rectangle of a decoded NV12 buffer, copied out tightly packed.
fn copy_planes(image: CfTypeRef, format: u32) -> Result<Nv12Frame, String> {
    // SAFETY: `image` is a live CVPixelBuffer for the handler's duration.
    let (got, cw, ch, planes) = unsafe {
        (
            CVPixelBufferGetPixelFormatType(image),
            CVPixelBufferGetWidth(image),
            CVPixelBufferGetHeight(image),
            CVPixelBufferGetPlaneCount(image),
        )
    };
    if got != format {
        return Err(format!("the decoder returned pixel format {got:#x}, not {format:#x}"));
    }
    if planes != 2 {
        return Err(format!("an NV12 buffer has {planes} planes, not 2"));
    }
    let (x0, y0, w, h) = picture_rect(image, (cw, ch))?;
    // SAFETY: as above.
    let dims = |p: usize| unsafe {
        (
            CVPixelBufferGetWidthOfPlane(image, p),
            CVPixelBufferGetHeightOfPlane(image, p),
        )
    };
    if dims(0) != (cw, ch) || dims(1) != (cw.div_ceil(2), ch.div_ceil(2)) {
        return Err(format!(
            "NV12 planes are {:?} and {:?} for a {cw}x{ch} buffer",
            dims(0),
            dims(1)
        ));
    }
    let lock = Lock::new(image)?;
    // SAFETY: the buffer is locked (`lock`), so the plane base addresses and
    // strides are valid until it drops at the end of this function.
    let (y_base, y_stride, uv_base, uv_stride) = unsafe {
        (
            CVPixelBufferGetBaseAddressOfPlane(image, 0),
            CVPixelBufferGetBytesPerRowOfPlane(image, 0),
            CVPixelBufferGetBaseAddressOfPlane(image, 1),
            CVPixelBufferGetBytesPerRowOfPlane(image, 1),
        )
    };
    if y_base.is_null() || uv_base.is_null() || y_stride < cw || uv_stride < cw {
        return Err(format!(
            "the planes have no readable rows (strides {y_stride} and {uv_stride} for {cw} pixels)"
        ));
    }
    let mut y = Vec::with_capacity(w * h);
    for row in 0..h {
        // SAFETY: the locked luma plane holds `ch >= y0 + h` rows of `y_stride >=
        // cw >= x0 + w` bytes; this reads `w` bytes of row `y0 + row` from column `x0`.
        y.extend_from_slice(unsafe { std::slice::from_raw_parts(y_base.add((y0 + row) * y_stride + x0), w) });
    }
    let mut uv = Vec::with_capacity(w * h / 2);
    for row in 0..h / 2 {
        // SAFETY: the chroma plane holds `ch / 2` rows of `uv_stride >= cw` bytes
        // (Cb, Cr pairs for `cw / 2` chroma columns); the picture starts at byte
        // `x0` (an even column, so a whole pair) of row `y0 / 2` and spans `w` bytes.
        uv.extend_from_slice(unsafe { std::slice::from_raw_parts(uv_base.add((y0 / 2 + row) * uv_stride + x0), w) });
    }
    drop(lock);
    Ok(Nv12Frame {
        width: w,
        height: h,
        full_range: format == FORMAT_420F,
        y,
        uv,
    })
}
