//! Decoding a track's compressed samples with VideoToolbox, to the decoder's own
//! planes: 8 or 10 bits, 4:2:0, 4:2:2 or 4:4:4 (the six bi-planar formats of each
//! depth in `CVPixelBuffer.h`; the stream's own, never a converted one).
//!
//! # Why not `AVAssetReader` for this
//!
//! `AVAssetReaderTrackOutput` can vend decoded `420v`/`420f` pixel buffers, and
//! their luma is the decoder's. Their chroma is not: against FFmpeg's decode of
//! the same file (and against the source of a lossless stream), many chroma
//! samples differ, by up to 45 levels at colour edges and by one or two in
//! gradients, for H.264 and HEVC, with deblocking off, on intra-only streams, and
//! with `AVVideoColorPropertiesKey` set. The reader resamples the chroma to the
//! `BottomLeft` location (measured: a stream tagged `BottomLeft` comes out
//! unchanged, one tagged `Left` is moved a quarter chroma row by a three-tap kernel
//! of about 0.06, 0.62, 0.31; `Center` moves in both directions; see "Why
//! AVAssetReader's chroma differs" in `docs/embedgemma2.md`). The same decoder
//! driven directly returns FFmpeg's bytes: FFmpeg's own
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
use super::video::{CmTime, NativeFrame, Samples};
use super::yuv::{Chroma, Layout};
use super::MediaError;

/// `kCFNumberFloat64Type`.
const CF_NUMBER_FLOAT64_TYPE: isize = 6;
/// `kCVPixelBufferLock_ReadOnly`.
const LOCK_READ_ONLY: u64 = 1;
/// `kVTDecodeInfo_FrameDropped`.
const INFO_FRAME_DROPPED: u32 = 1 << 1;

/// The bi-planar pixel format of `CVPixelBuffer.h` that holds a stream of this
/// layout in the stream's own range. 8-bit: `kCVPixelFormatType_420YpCbCr8BiPlanar`
/// `VideoRange`/`FullRange` ('420v', '420f'), the 4:2:2 ('422v', '422f') and 4:4:4
/// ('444v', '444f') kin. 10-bit (`420YpCbCr10BiPlanar...`, each sample in the
/// 10 most significant bits of a 16-bit word): 'x420', 'xf20', 'x422', 'xf22',
/// 'x444', 'xf44'.
pub(super) fn pixel_format(layout: Layout, full_range: bool) -> u32 {
    let code: &[u8; 4] = match (layout.depth, layout.chroma, full_range) {
        (8, Chroma::C420, false) => b"420v",
        (8, Chroma::C420, true) => b"420f",
        (8, Chroma::C422, false) => b"422v",
        (8, Chroma::C422, true) => b"422f",
        (8, Chroma::C444, false) => b"444v",
        (8, Chroma::C444, true) => b"444f",
        (_, Chroma::C420, false) => b"x420",
        (_, Chroma::C420, true) => b"xf20",
        (_, Chroma::C422, false) => b"x422",
        (_, Chroma::C422, true) => b"xf22",
        (_, Chroma::C444, false) => b"x444",
        (_, Chroma::C444, true) => b"xf44",
    };
    u32::from_be_bytes(*code)
}

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
    frames: Vec<Option<NativeFrame>>,
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
    layout: Layout,
    full_range: bool,
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
    /// decoding to the stream's own pixel format ([`pixel_format`] of `layout`
    /// and `full_range`). `pts` are the presentation times of the track's frames
    /// in presentation order, `wanted` which of them to keep.
    pub(super) fn new(
        first: CfTypeRef,
        layout: Layout,
        full_range: bool,
        pts: Vec<i64>,
        wanted: Vec<bool>,
    ) -> Result<Decoder, MediaError> {
        let format = pixel_format(layout, full_range);
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
            layout,
            full_range,
        })
    }

    /// Decode one compressed sample. Samples must come in decode order and share
    /// the first one's format description.
    pub(super) fn decode(&mut self, sample: CfTypeRef) -> Result<(), MediaError> {
        let (layout, full_range) = (self.layout, self.full_range);
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
                        guard.frames[index] = Some(copy_planes(image, layout, full_range)?);
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
    pub(super) fn finish(self) -> Result<Vec<Option<NativeFrame>>, MediaError> {
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
fn picture_rect(
    image: CfTypeRef,
    coded: (usize, usize),
    (sub_w, sub_h): (usize, usize),
) -> Result<(usize, usize, usize, usize), String> {
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
    // The rectangle must start and end on chroma samples: H.264 and HEVC crop in
    // units of the chroma sample, so a conforming stream does.
    if x0 + w > coded.0 || y0 + h > coded.1 || x0 % sub_w != 0 || y0 % sub_h != 0 || w % sub_w != 0 || h % sub_h != 0 {
        return Err(format!(
            "the clean aperture {w}x{h} at ({x0}, {y0}) does not fit a {}x{} buffer whose chroma is subsampled by {sub_w}x{sub_h}",
            coded.0, coded.1
        ));
    }
    Ok((x0, y0, w, h))
}

/// Chroma subsampling factors `(horizontal, vertical)` of a layout.
fn subsampling(chroma: Chroma) -> (usize, usize) {
    match chroma {
        Chroma::C420 => (2, 2),
        Chroma::C422 => (2, 1),
        Chroma::C444 => (1, 1),
    }
}

/// A rectangle of one locked plane as samples: `rows` rows of `per_row` samples
/// starting at row `row0`, sample `col0`. 8-bit planes are bytes; deeper ones are
/// 16-bit words with the sample in the high bits (`depth` of them), returned
/// right-aligned, and a word with a non-zero low part is an error.
///
/// # Safety
/// The plane must be locked, hold at least `row0 + rows` rows of `stride` bytes,
/// and each row at least `col0 + per_row` samples.
unsafe fn read_plane(
    base: *const u8,
    stride: usize,
    (row0, rows): (usize, usize),
    (col0, per_row): (usize, usize),
    depth: u8,
) -> Result<Samples, String> {
    if depth == 8 {
        let mut out = Vec::with_capacity(rows * per_row);
        for r in 0..rows {
            // SAFETY: the caller guarantees the row holds `col0 + per_row` bytes.
            out.extend_from_slice(std::slice::from_raw_parts(
                base.add((row0 + r) * stride + col0),
                per_row,
            ));
        }
        return Ok(Samples::U8(out));
    }
    let shift = 16 - u32::from(depth);
    let mut out = Vec::with_capacity(rows * per_row);
    let mut low = 0u16;
    for r in 0..rows {
        let row = base.add((row0 + r) * stride + col0 * 2);
        for i in 0..per_row {
            // SAFETY: the caller guarantees `col0 + per_row` words in the row; the
            // read is unaligned-safe.
            let word = row.add(i * 2).cast::<u16>().read_unaligned();
            low |= word & ((1 << shift) - 1);
            out.push(word >> shift);
        }
    }
    if low != 0 {
        return Err(format!(
            "a {depth}-bit sample word has bits set below its {depth} high bits ({low:#06x})"
        ));
    }
    Ok(Samples::U16(out))
}

/// The picture rectangle of a decoded bi-planar buffer, copied out tightly packed
/// (planes `Y` and interleaved `CbCr`), deeper-than-8-bit samples right-aligned.
fn copy_planes(image: CfTypeRef, layout: Layout, full_range: bool) -> Result<NativeFrame, String> {
    let format = pixel_format(layout, full_range);
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
        return Err(format!("a bi-planar buffer has {planes} planes, not 2"));
    }
    let (sub_w, sub_h) = subsampling(layout.chroma);
    let (x0, y0, w, h) = picture_rect(image, (cw, ch), (sub_w, sub_h))?;
    // SAFETY: as above.
    let dims = |p: usize| unsafe {
        (
            CVPixelBufferGetWidthOfPlane(image, p),
            CVPixelBufferGetHeightOfPlane(image, p),
        )
    };
    let (ccw, cch) = (cw.div_ceil(sub_w), ch.div_ceil(sub_h));
    if dims(0) != (cw, ch) || dims(1) != (ccw, cch) {
        return Err(format!(
            "the planes are {:?} and {:?} for a {cw}x{ch} {:?} buffer",
            dims(0),
            dims(1),
            layout.chroma
        ));
    }
    let bytes = if layout.depth == 8 { 1 } else { 2 };
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
    if y_base.is_null() || uv_base.is_null() || y_stride < cw * bytes || uv_stride < 2 * ccw * bytes {
        return Err(format!(
            "the planes have no readable rows (strides {y_stride} and {uv_stride} for {cw} pixels)"
        ));
    }
    // SAFETY: the luma plane holds `ch >= y0 + h` rows of `y_stride >= cw * bytes
    // >= (x0 + w) * bytes` bytes (the rectangle was checked to lie in the buffer);
    // the chroma plane holds `cch` rows of `uv_stride >= 2 * ccw * bytes` bytes,
    // `ccw` pairs each, and the rectangle starts on a chroma sample (`picture_rect`),
    // so it spans `h / sub_h` rows of `w / sub_w` pairs from pair `x0 / sub_w` of
    // row `y0 / sub_h`.
    let (y, uv) = unsafe {
        (
            read_plane(y_base, y_stride, (y0, h), (x0, w), layout.depth)?,
            read_plane(
                uv_base,
                uv_stride,
                (y0 / sub_h, h / sub_h),
                (2 * (x0 / sub_w), 2 * (w / sub_w)),
                layout.depth,
            )?,
        )
    };
    drop(lock);
    Ok(NativeFrame {
        width: w,
        height: h,
        layout,
        full_range,
        y,
        uv,
    })
}
