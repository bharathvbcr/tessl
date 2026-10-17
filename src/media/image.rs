//! Image decoding with ImageIO and CoreGraphics.
//!
//! PNG, JPEG, GIF (first frame), HEIC, TIFF, BMP and whatever else ImageIO
//! reads become 8-bit RGB or RGBA, row-major HWC, ready for
//! [`crate::embedgemma2_pre::RawImage`].
//!
//! # What reaches the model, and what this matches
//!
//! sentence-transformers (6.1) with transformers 5.19 does not open a path
//! with PIL. For the `feature-extraction` task the string reaches the image
//! processor's `fetch_images`, which calls
//! `torchvision.io.decode_image(path, mode=ImageReadMode.RGB)`
//! (libpng/libjpeg-turbo, no colour management, no EXIF transpose). A PIL image
//! given directly, and `transformers.image_utils.load_image`, go through
//! `ImageOps.exif_transpose` and `convert("RGB")` instead. The defaults here
//! follow the first (no EXIF transpose, no colour conversion); the second is
//! [`ExifOrientation::Apply`].
//!
//! # Three paths
//!
//! * **JPEG** ([`DecodePath::Jpeg`]): [`super::jpeg`], a pure-Rust decoder whose
//!   RGB is bit-identical to libjpeg-turbo's, so to what the reference path
//!   returns (ImageIO's JPEG decoder is not libjpeg-turbo). CMYK and YCCK JPEGs
//!   are converted to RGB the way torchvision does it. HEIC and every other
//!   format below stay with ImageIO.
//! * **Direct** (every case measured): the decoded `CGImage` already holds the
//!   file's samples as 8- or 16-bit gray or RGB with or without alpha, or as an
//!   8-bit palette. The samples are copied out of the image's data provider
//!   (`CGDataProviderCopyData`), honouring its row stride, byte order and alpha
//!   placement, with no colour conversion and no resampling. Alpha is returned
//!   as stored, non-premultiplied wherever ImageIO decodes it so (PNG, TIFF,
//!   GIF, BMP did); a premultiplied image is un-premultiplied by rounding,
//!   which cannot recover a colour under alpha 0. This is PIL's and libpng's
//!   result: bytes equal.
//! * **sRGB context** (CMYK, float, packed 5-6-5, unusual depths, or
//!   [`ColorHandling::Srgb`] on an image that is not sRGB): the image is drawn
//!   into a premultiplied RGBA8 bitmap context in the device-independent sRGB
//!   space, so Quartz does the colour conversion, and the result is
//!   un-premultiplied. Exact for opaque pixels, within one level otherwise, and
//!   different from PIL by whatever the profile conversion moves.
//!
//! [`ImageInfo::path`] records which ran.

use std::path::Path;

use super::cf::{self, CfRef, CfTypeRef};
use super::{jpeg, orient, MediaError};
use crate::embedgemma2_pre::{RawImage, MAX_IMAGE_PIXELS};

// ---------------------------------------------------------------------------
// FFI: ImageIO (CGImageSource.h, CGImageProperties.h) and CoreGraphics
// (CGImage.h, CGColorSpace.h, CGDataProvider.h, CGBitmapContext.h,
// CGContext.h, CGGeometry.h), macOS SDK.
// ---------------------------------------------------------------------------

/// `CGRect` (`CGGeometry.h`): origin and size, `CGFloat` being `double`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CgRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// `CGImageAlphaInfo` (`CGImage.h`).
const ALPHA_NONE: u32 = 0;
const ALPHA_PREMULTIPLIED_LAST: u32 = 1;
const ALPHA_PREMULTIPLIED_FIRST: u32 = 2;
const ALPHA_LAST: u32 = 3;
const ALPHA_FIRST: u32 = 4;
const ALPHA_NONE_SKIP_LAST: u32 = 5;
const ALPHA_NONE_SKIP_FIRST: u32 = 6;
/// `CGBitmapInfo` masks and byte orders (`CGImage.h`).
const BITMAP_ALPHA_MASK: u32 = 0x1F;
const BITMAP_COMPONENT_MASK: u32 = 0xF00;
const BITMAP_BYTE_ORDER_MASK: u32 = 0x7000;
const BITMAP_PIXEL_FORMAT_MASK: u32 = 0xF_0000;
const BYTE_ORDER_DEFAULT: u32 = 0;
const BYTE_ORDER_16_LITTLE: u32 = 1 << 12;
const BYTE_ORDER_32_LITTLE: u32 = 2 << 12;
const BYTE_ORDER_16_BIG: u32 = 3 << 12;
const BYTE_ORDER_32_BIG: u32 = 4 << 12;
/// `CGColorSpaceModel` (`CGColorSpace.h`).
const MODEL_MONOCHROME: i32 = 0;
const MODEL_RGB: i32 = 1;
const MODEL_CMYK: i32 = 2;
const MODEL_INDEXED: i32 = 5;
/// `CGImageSourceStatus::kCGImageStatusComplete` (`CGImageSource.h`).
const STATUS_COMPLETE: i32 = 0;

#[link(name = "ImageIO", kind = "framework")]
extern "C" {
    static kCGImagePropertyPixelWidth: CfTypeRef;
    static kCGImagePropertyPixelHeight: CfTypeRef;
    static kCGImagePropertyOrientation: CfTypeRef;

    fn CGImageSourceCreateWithData(data: CfTypeRef, options: CfTypeRef) -> CfTypeRef;
    fn CGImageSourceGetType(isrc: CfTypeRef) -> CfTypeRef;
    fn CGImageSourceGetCount(isrc: CfTypeRef) -> usize;
    fn CGImageSourceGetStatusAtIndex(isrc: CfTypeRef, index: usize) -> i32;
    fn CGImageSourceCopyPropertiesAtIndex(isrc: CfTypeRef, index: usize, options: CfTypeRef) -> CfTypeRef;
    fn CGImageSourceCreateImageAtIndex(isrc: CfTypeRef, index: usize, options: CfTypeRef) -> CfTypeRef;
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    static kCGColorSpaceSRGB: CfTypeRef;

    fn CGImageGetWidth(image: CfTypeRef) -> usize;
    fn CGImageGetHeight(image: CfTypeRef) -> usize;
    fn CGImageGetBitsPerComponent(image: CfTypeRef) -> usize;
    fn CGImageGetBitsPerPixel(image: CfTypeRef) -> usize;
    fn CGImageGetBytesPerRow(image: CfTypeRef) -> usize;
    fn CGImageGetColorSpace(image: CfTypeRef) -> CfTypeRef;
    fn CGImageGetAlphaInfo(image: CfTypeRef) -> u32;
    fn CGImageGetBitmapInfo(image: CfTypeRef) -> u32;
    fn CGImageGetDataProvider(image: CfTypeRef) -> CfTypeRef;
    fn CGDataProviderCopyData(provider: CfTypeRef) -> CfTypeRef;

    fn CGColorSpaceGetModel(space: CfTypeRef) -> i32;
    fn CGColorSpaceGetNumberOfComponents(space: CfTypeRef) -> usize;
    fn CGColorSpaceGetColorTableCount(space: CfTypeRef) -> usize;
    fn CGColorSpaceGetColorTable(space: CfTypeRef, table: *mut u8);
    fn CGColorSpaceGetBaseColorSpace(space: CfTypeRef) -> CfTypeRef;
    fn CGColorSpaceCopyName(space: CfTypeRef) -> CfTypeRef;
    fn CGColorSpaceCreateWithName(name: CfTypeRef) -> CfTypeRef;

    fn CGBitmapContextCreate(
        data: *mut std::ffi::c_void,
        width: usize,
        height: usize,
        bits_per_component: usize,
        bytes_per_row: usize,
        space: CfTypeRef,
        bitmap_info: u32,
    ) -> CfTypeRef;
    fn CGContextDrawImage(c: CfTypeRef, rect: CgRect, image: CfTypeRef);
}

// ---------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------

/// The channel layout of the decoded bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    /// Red, green, blue; alpha dropped without compositing (PIL's
    /// `convert("RGB")`).
    Rgb,
    /// Red, green, blue, alpha; 255 where the image has none (PIL's
    /// `convert("RGBA")`).
    Rgba,
}

/// Whether to apply the EXIF orientation tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExifOrientation {
    /// Return the stored pixels, as `torchvision.io.decode_image(path)` does,
    /// which is what sentence-transformers runs on a path.
    Ignore,
    /// Transpose as `PIL.ImageOps.exif_transpose` does, which is what
    /// `transformers.image_utils.load_image` and a PIL image input do.
    Apply,
}

/// How to treat the image's colour space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorHandling {
    /// Take the stored samples as they are, ignoring any embedded profile: PIL,
    /// libpng and libjpeg-turbo, so also `torchvision`.
    Native,
    /// Convert to sRGB (Quartz colour management) when the image's colour
    /// space is not already sRGB.
    Srgb,
}

/// What 16-bit gray (no alpha) becomes in 8 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gray16 {
    /// `min(v, 255)`: what Pillow 12.3 returns for `convert("RGB")` of an
    /// `I;16` image. It turns every sample above 255 white, so a 16-bit gray
    /// PNG usually comes out as a blown-out image; it is the default only
    /// because it is the reference's behaviour.
    PilClip,
    /// `v >> 8`: the high byte, which is also what Pillow does for 16-bit RGB,
    /// RGBA and gray+alpha.
    HighByte,
}

/// Options for [`decode_image`] and [`decode_image_file`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageOptions {
    pub layout: ImageLayout,
    pub orientation: ExifOrientation,
    pub color: ColorHandling,
    pub gray16: Gray16,
    /// Refuse images with more pixels than this, before decoding them.
    pub max_pixels: usize,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            layout: ImageLayout::Rgb,
            orientation: ExifOrientation::Ignore,
            color: ColorHandling::Native,
            gray16: Gray16::PilClip,
            max_pixels: MAX_IMAGE_PIXELS,
        }
    }
}

/// The largest image file [`decode_image_file`] reads.
pub const MAX_IMAGE_FILE_BYTES: u64 = 1 << 30;

/// Which of the two extraction paths ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePath {
    /// The image's samples copied out as stored.
    Direct,
    /// Drawn into an sRGB RGBA8 bitmap context.
    SrgbContext,
    /// Decoded by [`super::jpeg`], this crate's libjpeg-turbo-exact decoder.
    Jpeg,
}

/// The colour model of the decoded image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorModel {
    Gray,
    Rgb,
    Cmyk,
    Indexed,
    Other,
}

/// What ImageIO reported about the image and what was done with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    /// The container's uniform type identifier, e.g. `public.png`.
    pub format: String,
    /// Bits per component of the decoded `CGImage`.
    pub bits_per_component: usize,
    pub color_model: ColorModel,
    /// The colour space's name when it has one (`kCGColorSpaceSRGB` ...).
    pub color_space: Option<String>,
    /// The EXIF orientation ImageIO reports (1 when there is none).
    pub exif_orientation: u32,
    /// Whether the image carried an alpha channel (premultiplied or not).
    pub has_alpha: bool,
    pub path: DecodePath,
}

/// A decoded image: `height * width * channels` bytes, row-major HWC.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: usize,
    pub height: usize,
    /// 3 ([`ImageLayout::Rgb`]) or 4 ([`ImageLayout::Rgba`]).
    pub channels: usize,
    pub data: Vec<u8>,
    pub info: ImageInfo,
}

impl DecodedImage {
    /// The bytes as the preprocessing module's input.
    pub fn raw(&self) -> RawImage<'_> {
        RawImage {
            width: self.width,
            height: self.height,
            channels: self.channels,
            data: &self.data,
        }
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode an image file.
///
/// # Errors
/// [`MediaError::Io`] when it cannot be read, [`MediaError::Empty`] for an empty
/// file, [`MediaError::Limit`] over [`MAX_IMAGE_FILE_BYTES`] or
/// `options.max_pixels`, and the errors of [`decode_image`].
pub fn decode_image_file(path: &Path, options: &ImageOptions) -> Result<DecodedImage, MediaError> {
    let bytes = super::read_file_bounded(path, MAX_IMAGE_FILE_BYTES, "image")?;
    decode_image(&bytes, options)
}

/// Decode an image from its encoded bytes (PNG, JPEG, GIF, HEIC, TIFF, BMP, ...).
/// Only the first image of a container (the first GIF frame, the primary HEIC
/// image) is decoded.
///
/// A JPEG (bytes starting `FF D8 FF`) is decoded by this crate's own decoder,
/// [`super::jpeg`], whose output is bit-identical to libjpeg-turbo's, which is
/// what `torchvision.io.decode_image` returns; [`DecodePath::Jpeg`] says so.
/// ImageIO's JPEG decoder is not libjpeg-turbo and differs by up to 43 levels on
/// 4:2:0 pictures. The one JPEG case that still goes to ImageIO is
/// [`ColorHandling::Srgb`], because colour management is Quartz's. Everything
/// else is decoded by ImageIO: see [`decode_image_imageio`].
///
/// # Errors
/// [`MediaError::Empty`] for no bytes, [`MediaError::Unrecognized`] when
/// the system does not know the format, [`MediaError::Limit`] when the image has
/// more than `options.max_pixels` pixels, [`MediaError::Decode`] when the data is
/// damaged or truncated, and [`MediaError::Unsupported`] for a JPEG kind the
/// native decoder names and refuses (12-bit, lossless, progressive with block
/// smoothing).
pub fn decode_image(bytes: &[u8], options: &ImageOptions) -> Result<DecodedImage, MediaError> {
    if jpeg::has_jpeg_signature(bytes) && options.color == ColorHandling::Native {
        return decode_jpeg_native(bytes, options);
    }
    decode_image_imageio(bytes, options)
}

/// The native JPEG decoder behind [`decode_image`].
fn decode_jpeg_native(bytes: &[u8], options: &ImageOptions) -> Result<DecodedImage, MediaError> {
    const WHAT: &str = "image";
    let img = jpeg::decode(bytes, options.max_pixels).map_err(|e| match e {
        jpeg::JpegError::Limit(detail) => MediaError::Limit { what: WHAT, detail },
        jpeg::JpegError::Unsupported(detail) => MediaError::Unsupported { what: WHAT, detail },
        other => MediaError::Decode {
            what: WHAT,
            detail: format!("public.jpeg: {other}"),
        },
    })?;
    let orientation = img.info.exif_orientation.unwrap_or(1);
    let (width, height) = (img.width, img.height);
    let (channels, mut data) = match options.layout {
        ImageLayout::Rgb => (3, img.rgb),
        ImageLayout::Rgba => {
            let mut rgba = Vec::with_capacity(width * height * 4);
            for px in img.rgb.chunks_exact(3) {
                rgba.extend_from_slice(px);
                rgba.push(255);
            }
            (4, rgba)
        }
    };
    let (mut out_w, mut out_h) = (width, height);
    if options.orientation == ExifOrientation::Apply && orientation != 1 {
        let (d, nw, nh) = orient::apply(orientation, std::mem::take(&mut data), width, height, channels);
        data = d;
        out_w = nw;
        out_h = nh;
    }
    let color_model = match img.info.color_space {
        jpeg::JpegColorSpace::Grayscale => ColorModel::Gray,
        jpeg::JpegColorSpace::Cmyk | jpeg::JpegColorSpace::Ycck => ColorModel::Cmyk,
        _ => ColorModel::Rgb,
    };
    Ok(DecodedImage {
        width: out_w,
        height: out_h,
        channels,
        data,
        info: ImageInfo {
            format: "public.jpeg".into(),
            bits_per_component: 8,
            color_model,
            color_space: None,
            exif_orientation: orientation,
            has_alpha: false,
            path: DecodePath::Jpeg,
        },
    })
}

/// Decode an image with ImageIO whatever its format (PNG, JPEG, GIF, HEIC,
/// TIFF, BMP, ...). [`decode_image`] sends JPEG elsewhere; this is the path for
/// everything else, and for comparing ImageIO's JPEG decode with the native one.
///
/// # Errors
/// [`MediaError::Empty`] for no bytes, [`MediaError::Unrecognized`] when
/// ImageIO does not know the format, [`MediaError::Limit`] when the image has
/// more than `options.max_pixels` pixels, and [`MediaError::Decode`] when the
/// data is damaged or truncated (ImageIO's status is anything but complete).
pub fn decode_image_imageio(bytes: &[u8], options: &ImageOptions) -> Result<DecodedImage, MediaError> {
    const WHAT: &str = "image";
    if bytes.is_empty() {
        return Err(MediaError::Empty { what: WHAT });
    }
    // ImageIO accepts some cut-short containers (TIFF, BMP, a PNG cut inside its
    // last CRC) as complete pictures with zeros where the data is missing.
    super::integrity::check_container(bytes).map_err(|detail| MediaError::Decode { what: WHAT, detail })?;
    let data = cf::data_from_bytes(bytes).ok_or_else(|| MediaError::Framework {
        api: "CFDataCreate",
        detail: format!("could not copy {} bytes", bytes.len()),
    })?;
    // SAFETY: `data` is a live CFData; a NULL options dictionary is allowed;
    // the result is +1 or NULL.
    let source = unsafe { CfRef::take(CGImageSourceCreateWithData(data.as_ptr(), std::ptr::null())) }
        .ok_or_else(|| MediaError::unrecognized(WHAT, "ImageIO makes no image source from these bytes"))?;
    // SAFETY: `source` is a live CGImageSource; Get functions return unowned
    // values valid while it lives.
    let count = unsafe { CGImageSourceGetCount(source.as_ptr()) };
    if count == 0 {
        return Err(MediaError::unrecognized(WHAT, "the source holds no images"));
    }
    // SAFETY: as above; CGImageSourceGetType returns NULL or an unowned CFString.
    let format = unsafe { cf::cf_string(CGImageSourceGetType(source.as_ptr())) }.unwrap_or_default();

    // The properties are read before the image is decoded, so a bound on the
    // size is enforced before ImageIO allocates the pixels.
    // SAFETY: `source` is live; the copy is +1 or NULL.
    let props = unsafe { CfRef::take(CGImageSourceCopyPropertiesAtIndex(source.as_ptr(), 0, std::ptr::null())) }
        .ok_or_else(|| decode_error(&source, &format, "no properties for the first image"))?;
    // SAFETY: `props` is a live CFDictionary and the keys are SDK constants.
    let (pw, ph, orientation) = unsafe {
        (
            cf::dict_i64(props.as_ptr(), kCGImagePropertyPixelWidth),
            cf::dict_i64(props.as_ptr(), kCGImagePropertyPixelHeight),
            cf::dict_i64(props.as_ptr(), kCGImagePropertyOrientation),
        )
    };
    let (Some(pw), Some(ph)) = (pw, ph) else {
        return Err(decode_error(&source, &format, "the header has no pixel size"));
    };
    let (pw, ph) = (usize::try_from(pw).unwrap_or(0), usize::try_from(ph).unwrap_or(0));
    if pw == 0 || ph == 0 {
        return Err(decode_error(&source, &format, "the image has no pixels"));
    }
    match pw.checked_mul(ph) {
        Some(p) if p <= options.max_pixels => {}
        _ => {
            return Err(MediaError::Limit {
                what: WHAT,
                detail: format!("a {ph}x{pw} image exceeds {} pixels", options.max_pixels),
            })
        }
    }
    let orientation = u32::try_from(orientation.unwrap_or(1))
        .ok()
        .filter(|o| (1..=8).contains(o))
        .unwrap_or(1);

    // SAFETY: `source` is live; the image is +1 or NULL.
    let image = unsafe { CfRef::take(CGImageSourceCreateImageAtIndex(source.as_ptr(), 0, std::ptr::null())) }
        .ok_or_else(|| decode_error(&source, &format, "ImageIO returned no image"))?;
    let img = image.as_ptr();
    // SAFETY: `img` is a live CGImage.
    let (w, h) = unsafe { (CGImageGetWidth(img), CGImageGetHeight(img)) };
    if (w, h) != (pw, ph) {
        return Err(decode_error(
            &source,
            &format,
            &format!("the header says {ph}x{pw} but the image is {h}x{w}"),
        ));
    }

    let probe = Probe::of(img);
    let want_direct = options.color == ColorHandling::Native || probe.is_srgb;
    let direct = if want_direct { probe.direct() } else { None };
    let (mut data, path) = match direct {
        Some(layout) => {
            let pixels = extract_direct(img, &layout, w, h, options)
                .ok_or_else(|| decode_error(&source, &format, "the image's pixel data is shorter than its size"))?;
            (pixels, DecodePath::Direct)
        }
        None => (draw_srgb(img, w, h, options.layout)?, DecodePath::SrgbContext),
    };
    // A truncated or damaged file is decoded as far as it goes; ImageIO says so
    // only through the status, which is final once the pixels have been read.
    // SAFETY: `source` is live.
    let status = unsafe { CGImageSourceGetStatusAtIndex(source.as_ptr(), 0) };
    if status != STATUS_COMPLETE {
        return Err(MediaError::Decode {
            what: WHAT,
            detail: format!("{format}: ImageIO status {status} ({})", status_name(status)),
        });
    }

    let channels = match options.layout {
        ImageLayout::Rgb => 3,
        ImageLayout::Rgba => 4,
    };
    let (mut out_w, mut out_h) = (w, h);
    if options.orientation == ExifOrientation::Apply && orientation != 1 {
        let (d, nw, nh) = orient::apply(orientation, std::mem::take(&mut data), w, h, channels);
        data = d;
        out_w = nw;
        out_h = nh;
    }
    Ok(DecodedImage {
        width: out_w,
        height: out_h,
        channels,
        data,
        info: ImageInfo {
            format,
            bits_per_component: probe.bpc,
            color_model: probe.color_model(),
            color_space: probe.space_name.clone(),
            exif_orientation: orientation,
            has_alpha: probe.has_alpha(),
            path,
        },
    })
}

fn status_name(status: i32) -> &'static str {
    match status {
        -5 => "unexpected end of file",
        -4 => "invalid data",
        -3 => "unknown type",
        -2 => "reading header",
        -1 => "incomplete",
        0 => "complete",
        _ => "unknown",
    }
}

/// A `MediaError::Decode` that names ImageIO's own account of what is wrong.
fn decode_error(source: &CfRef, format: &str, why: &str) -> MediaError {
    // SAFETY: `source` is a live CGImageSource.
    let status = unsafe { CGImageSourceGetStatusAtIndex(source.as_ptr(), 0) };
    MediaError::Decode {
        what: "image",
        detail: format!("{format}: {why} (ImageIO status {status}, {})", status_name(status)),
    }
}

// ---------------------------------------------------------------------------
// What the decoded CGImage looks like
// ---------------------------------------------------------------------------

/// The facts about a `CGImage` that choose the extraction path.
struct Probe {
    bpc: usize,
    bpp: usize,
    alpha_info: u32,
    bitmap_info: u32,
    /// `CGColorSpaceGetModel`, or -1 when the image has no colour space.
    model: i32,
    components: usize,
    space_name: Option<String>,
    is_srgb: bool,
    /// For an indexed image: its RGB palette (`entries * 3` bytes), if the base
    /// space is RGB.
    palette: Option<Vec<u8>>,
}

impl Probe {
    fn of(img: CfTypeRef) -> Probe {
        // SAFETY: `img` is a live CGImage; every Get below returns a value
        // owned by the image (the colour space) or a plain integer.
        unsafe {
            let space = CGImageGetColorSpace(img);
            let (model, components, name, is_srgb, palette) = if space.is_null() {
                (-1, 0, None, false, None)
            } else {
                let model = CGColorSpaceGetModel(space);
                let name_ref = CfRef::take(CGColorSpaceCopyName(space));
                let name = name_ref.as_ref().and_then(|n| cf::cf_string(n.as_ptr()));
                let is_srgb = name_ref
                    .as_ref()
                    .is_some_and(|n| cf::CFEqual(n.as_ptr(), kCGColorSpaceSRGB) != 0);
                let palette = if model == MODEL_INDEXED {
                    let base = CGColorSpaceGetBaseColorSpace(space);
                    let entries = CGColorSpaceGetColorTableCount(space);
                    if !base.is_null() && CGColorSpaceGetModel(base) == MODEL_RGB && (1..=256).contains(&entries) {
                        let mut table = vec![0u8; entries * 3];
                        CGColorSpaceGetColorTable(space, table.as_mut_ptr());
                        Some(table)
                    } else {
                        None
                    }
                } else {
                    None
                };
                (model, CGColorSpaceGetNumberOfComponents(space), name, is_srgb, palette)
            };
            Probe {
                bpc: CGImageGetBitsPerComponent(img),
                bpp: CGImageGetBitsPerPixel(img),
                alpha_info: CGImageGetAlphaInfo(img),
                bitmap_info: CGImageGetBitmapInfo(img),
                model,
                components,
                space_name: name,
                is_srgb,
                palette,
            }
        }
    }

    fn color_model(&self) -> ColorModel {
        match self.model {
            MODEL_MONOCHROME => ColorModel::Gray,
            MODEL_RGB => ColorModel::Rgb,
            MODEL_CMYK => ColorModel::Cmyk,
            MODEL_INDEXED => ColorModel::Indexed,
            _ => ColorModel::Other,
        }
    }

    fn has_alpha(&self) -> bool {
        matches!(
            self.alpha_info,
            ALPHA_PREMULTIPLIED_LAST | ALPHA_PREMULTIPLIED_FIRST | ALPHA_LAST | ALPHA_FIRST
        )
    }

    /// The direct layout, or `None` when this image needs the sRGB context.
    fn direct(&self) -> Option<Direct> {
        // Float components, packed 565/555/101010 pixels and anything but the
        // plain integer layout go through the context.
        if self.bitmap_info & BITMAP_COMPONENT_MASK != 0 || self.bitmap_info & BITMAP_PIXEL_FORMAT_MASK != 0 {
            return None;
        }
        let alpha = self.bitmap_info & BITMAP_ALPHA_MASK;
        let order = self.bitmap_info & BITMAP_BYTE_ORDER_MASK;
        if self.model == MODEL_INDEXED {
            let palette = self.palette.clone()?;
            // One 8-bit index per pixel; transparency was already expanded by
            // ImageIO into an RGBA image, so an indexed image here is opaque.
            return (self.bpc == 8 && self.bpp == 8 && alpha == ALPHA_NONE).then_some(Direct::Indexed(palette));
        }
        let color_comps = match (self.model, self.components) {
            (MODEL_MONOCHROME, 1) => 1,
            (MODEL_RGB, 3) => 3,
            _ => return None,
        };
        let (alpha_pos, premultiplied, pad) = match alpha {
            ALPHA_NONE => (None, false, false),
            ALPHA_NONE_SKIP_LAST => (None, false, true),
            ALPHA_NONE_SKIP_FIRST => (None, false, true),
            ALPHA_LAST => (Some(AlphaPos::Last), false, false),
            ALPHA_FIRST => (Some(AlphaPos::First), false, false),
            ALPHA_PREMULTIPLIED_LAST => (Some(AlphaPos::Last), true, false),
            ALPHA_PREMULTIPLIED_FIRST => (Some(AlphaPos::First), true, false),
            _ => return None,
        };
        let skip_first = alpha == ALPHA_NONE_SKIP_FIRST;
        let comps = color_comps + usize::from(alpha_pos.is_some() || pad);
        if !(self.bpc == 8 || self.bpc == 16) || self.bpp != comps * self.bpc {
            return None;
        }
        let bytes_per_pixel = self.bpp / 8;
        // Memory position of logical component `i` (logical order is the one
        // the alpha info names: R G B A, A R G B, G A, A G, or R G B X ...).
        let reversed = match (self.bpc, order) {
            (8, BYTE_ORDER_DEFAULT | BYTE_ORDER_32_BIG) => false,
            (8, BYTE_ORDER_32_LITTLE) if bytes_per_pixel == 4 => true,
            (8, BYTE_ORDER_16_LITTLE) if bytes_per_pixel == 2 => true,
            (8, BYTE_ORDER_16_BIG) if bytes_per_pixel == 2 => false,
            (8, _) if bytes_per_pixel == 3 && order == BYTE_ORDER_DEFAULT => false,
            (16, BYTE_ORDER_16_LITTLE | BYTE_ORDER_16_BIG) => false,
            _ => return None,
        };
        let little_endian_samples = self.bpc == 16 && order == BYTE_ORDER_16_LITTLE;
        Some(Direct::Samples(SampleLayout {
            color_comps,
            comps,
            bytes_per_comp: self.bpc / 8,
            bytes_per_pixel,
            reversed,
            little_endian_samples,
            alpha_first: alpha_pos == Some(AlphaPos::First),
            has_alpha: alpha_pos.is_some(),
            premultiplied,
            skip_first,
        }))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AlphaPos {
    First,
    Last,
}

/// How to read samples out of the image's data.
enum Direct {
    /// Gray or RGB samples, with an optional alpha or padding component.
    Samples(SampleLayout),
    /// 8-bit indices into an RGB palette.
    Indexed(Vec<u8>),
}

struct SampleLayout {
    color_comps: usize,
    /// Components per pixel, alpha or padding included.
    comps: usize,
    bytes_per_comp: usize,
    bytes_per_pixel: usize,
    /// 8-bit components stored in reverse of logical order (32Little pixels).
    reversed: bool,
    /// 16-bit components are little-endian (else big-endian).
    little_endian_samples: bool,
    alpha_first: bool,
    has_alpha: bool,
    premultiplied: bool,
    /// A padding component comes first (XRGB) rather than last (RGBX).
    skip_first: bool,
}

impl SampleLayout {
    /// The component at memory component position `m` of a pixel, as a
    /// 16-bit value (8-bit samples are not widened: `max` says the range).
    #[inline]
    fn sample(&self, px: &[u8], logical: usize) -> u16 {
        let m = if self.reversed {
            self.comps - 1 - logical
        } else {
            logical
        };
        let at = m * self.bytes_per_comp;
        if self.bytes_per_comp == 1 {
            u16::from(px[at])
        } else if self.little_endian_samples {
            u16::from_le_bytes([px[at], px[at + 1]])
        } else {
            u16::from_be_bytes([px[at], px[at + 1]])
        }
    }
}

/// Copy the image's samples out as 8-bit `options.layout` pixels. `None` when
/// the data is shorter than the image's rows.
fn extract_direct(img: CfTypeRef, direct: &Direct, w: usize, h: usize, options: &ImageOptions) -> Option<Vec<u8>> {
    // SAFETY: `img` is a live CGImage; the provider is owned by it and the copy
    // is +1 or NULL.
    let data = unsafe { CfRef::take(CGDataProviderCopyData(CGImageGetDataProvider(img))) }?;
    // SAFETY: `data` is a live CFData.
    let bytes = unsafe { cf::data_bytes(&data) };
    // SAFETY: `img` is a live CGImage.
    let bpr = unsafe { CGImageGetBytesPerRow(img) };
    let channels = match options.layout {
        ImageLayout::Rgb => 3,
        ImageLayout::Rgba => 4,
    };
    let mut out = vec![0u8; w.checked_mul(h)?.checked_mul(channels)?];
    match direct {
        Direct::Indexed(palette) => {
            let entries = palette.len() / 3;
            if bpr < w || bytes.len() < bpr * (h - 1) + w {
                return None;
            }
            for y in 0..h {
                let row = &bytes[y * bpr..y * bpr + w];
                let dst = &mut out[y * w * channels..(y + 1) * w * channels];
                for (x, &idx) in row.iter().enumerate() {
                    // An index past the table is clamped to the last entry, as
                    // a palette lookup with a short table would be by libpng.
                    let i = usize::from(idx).min(entries - 1) * 3;
                    let d = &mut dst[x * channels..(x + 1) * channels];
                    d[..3].copy_from_slice(&palette[i..i + 3]);
                    if channels == 4 {
                        d[3] = 255;
                    }
                }
            }
        }
        Direct::Samples(l) => {
            let row_bytes = w * l.bytes_per_pixel;
            if bpr < row_bytes || bytes.len() < bpr * (h - 1) + row_bytes {
                return None;
            }
            let wide = l.bytes_per_comp == 2;
            let max: u32 = if wide { 65535 } else { 255 };
            // Logical component indices.
            let (ci, ai) = match (l.has_alpha, l.alpha_first) {
                (true, true) => (1, 0),
                (true, false) => (0, l.color_comps),
                (false, _) => (usize::from(l.skip_first), 0),
            };
            let reduce = |v: u32, is_gray_no_alpha: bool| -> u8 {
                if !wide {
                    v as u8
                } else if is_gray_no_alpha && options.gray16 == Gray16::PilClip {
                    v.min(255) as u8
                } else {
                    (v >> 8) as u8
                }
            };
            let gray_no_alpha = l.color_comps == 1 && !l.has_alpha;
            for y in 0..h {
                let row = &bytes[y * bpr..y * bpr + row_bytes];
                let dst = &mut out[y * w * channels..(y + 1) * w * channels];
                for x in 0..w {
                    let px = &row[x * l.bytes_per_pixel..(x + 1) * l.bytes_per_pixel];
                    let alpha = if l.has_alpha { u32::from(l.sample(px, ai)) } else { max };
                    let mut c = [0u32; 3];
                    for (k, slot) in c.iter_mut().enumerate().take(l.color_comps) {
                        *slot = u32::from(l.sample(px, ci + k));
                    }
                    if l.color_comps == 1 {
                        c[1] = c[0];
                        c[2] = c[0];
                    }
                    if l.premultiplied {
                        for v in c.iter_mut() {
                            *v = (*v * max + alpha / 2).checked_div(alpha).map_or(0, |q| q.min(max));
                        }
                    }
                    let d = &mut dst[x * channels..(x + 1) * channels];
                    d[0] = reduce(c[0], gray_no_alpha);
                    d[1] = reduce(c[1], gray_no_alpha);
                    d[2] = reduce(c[2], gray_no_alpha);
                    if channels == 4 {
                        d[3] = reduce(alpha, false);
                    }
                }
            }
        }
    }
    Some(out)
}

/// Draw the image into a premultiplied RGBA8 sRGB bitmap context and return
/// straight `layout` pixels.
fn draw_srgb(img: CfTypeRef, w: usize, h: usize, layout: ImageLayout) -> Result<Vec<u8>, MediaError> {
    // SAFETY: kCGColorSpaceSRGB is an SDK constant; the result is +1 or NULL.
    let space =
        unsafe { CfRef::take(CGColorSpaceCreateWithName(kCGColorSpaceSRGB)) }.ok_or_else(|| MediaError::Framework {
            api: "CGColorSpaceCreateWithName",
            detail: "no sRGB colour space".into(),
        })?;
    let row = w.checked_mul(4).ok_or(MediaError::Limit {
        what: "image",
        detail: "row size overflows".into(),
    })?;
    let mut buf = vec![
        0u8;
        row.checked_mul(h).ok_or(MediaError::Limit {
            what: "image",
            detail: "image size overflows".into(),
        })?
    ];
    // SAFETY: `buf` has `row * h` bytes and outlives the context, which is
    // released (dropped) before `buf` is read; the space is live; 8 bits per
    // component with premultiplied-last RGBA is a supported context format.
    let ctx = unsafe {
        CfRef::take(CGBitmapContextCreate(
            buf.as_mut_ptr().cast(),
            w,
            h,
            8,
            row,
            space.as_ptr(),
            ALPHA_PREMULTIPLIED_LAST,
        ))
    }
    .ok_or_else(|| MediaError::Framework {
        api: "CGBitmapContextCreate",
        detail: format!("could not make a {w}x{h} sRGB context"),
    })?;
    let rect = CgRect {
        x: 0.0,
        y: 0.0,
        width: w as f64,
        height: h as f64,
    };
    // SAFETY: the context and image are live; the rect is the image's own size,
    // so no scaling happens.
    unsafe { CGContextDrawImage(ctx.as_ptr(), rect, img) };
    drop(ctx);
    let channels = if layout == ImageLayout::Rgb { 3 } else { 4 };
    let mut out = vec![0u8; w * h * channels];
    for (src, dst) in buf.chunks_exact(4).zip(out.chunks_exact_mut(channels)) {
        let a = u32::from(src[3]);
        for k in 0..3 {
            dst[k] = match a {
                0 => 0,
                255 => src[k],
                _ => ((u32::from(src[k]) * 255 + a / 2) / a).min(255) as u8,
            };
        }
        if channels == 4 {
            dst[3] = src[3];
        }
    }
    Ok(out)
}
