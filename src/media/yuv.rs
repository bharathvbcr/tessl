//! YCbCr to RGB24, byte for byte what FFmpeg's libswscale computes for a
//! same-size `yuv4xxp` (8 or 10 bit) to `rgb24` conversion.
//!
//! This is the conversion the reference loader of EmbeddingGemma 2 runs.
//! `transformers` decodes a video file with torchcodec, and torchcodec turns
//! each decoded frame into RGB with libswscale (the frame's own
//! colorspace and range, no resize) or with the filter graph `format=rgb24`,
//! whose automatically inserted scaler is the same library. The result is a
//! function of the samples, the matrix, the range and (for one class, below)
//! the scaler's flags.
//!
//! # The converters
//!
//! libswscale reaches RGB through five different routines depending on the
//! sample format. [`to_rgb24`] picks the one the format takes and each is
//! checked against the `ffmpeg` binary (see "Parity" below):
//!
//! | input | routine | notes |
//! |---|---|---|
//! | 8-bit 4:2:0 and 4:2:2 | table converter, or on aarch64 the NEON one for widths that are a multiple of 16 and even heights | chroma replicated over 2 luma samples (and 2 rows for 4:2:0); [`Flavor`] |
//! | 8-bit 4:4:4 | direct: 13-bit coefficients in 32-bit integers | no table, no NEON; no flavor |
//! | 10-bit 4:4:4 | the same arithmetic at ten bits | |
//! | 10-bit 4:2:2 | luma and chroma rounded to 8 bits (`(v + 2) >> 2`), then the table converter | luma may reach 256, chroma is clipped to 255 |
//! | 10-bit 4:2:0 | as 4:2:2 after a vertical chroma filter (the scaler resamples chroma rows to luma rows) | [`ChromaFilter`]: the scaler flag decides the filter |
//!
//! Two arithmetics are unlike what one would write from the specifications and
//! are reproduced because the reference has them: the 4:4:4 sums are 32-bit
//! integers at 22 fractional bits that **wrap** (a blue value of 512 or more
//! comes out 0, not 255), and the sub-sampled 10-bit paths quantise to 8 bits
//! before the colour matrix.
//!
//! # Parity
//!
//! Every converter is held to the `ffmpeg` binary (9.0.2, aarch64) by
//! `tests/media_yuv_exact.rs` (8-bit 4:2:0) and `tests/media_yuv_formats.rs` (the
//! other formats). Their `--ignored` tests run `ffmpeg` live and compare every
//! `(Y, Cb, Cr)` triple of 8 bits (all 2^24, five matrices, both ranges, 4:2:0,
//! 4:2:2 and 4:4:4), sampled triples of 10 bits for 4:4:4 and 4:2:2, and the 10-bit
//! 4:2:0 filters at many heights, and the committed goldens hold random planes of
//! every format.
//!
//! # Where the numbers come from
//!
//! The four 16.16 chroma coefficients of a matrix are
//! `round(k * 65536 * 255 / 224)` for `k` in `2(1-Kr)`, `2(1-Kb)`,
//! `2Kb(1-Kb)/Kg` and `2Kr(1-Kr)/Kg` ([`Matrix::coefficients`]), with the `Kr`
//! and `Kb` of the matrix's public definition (ITU-R BT.709, BT.601,
//! BT.2020, FCC, SMPTE 240M). Both converters work from the 13-bit
//! rounding of those, `(c + 4) >> 3`.
//!
//! # Provenance (licensing)
//!
//! libswscale is LGPL-2.1+; this module is not a port of it. No libswscale
//! source file was read while writing it (none is installed here; the Homebrew
//! build ships only the public headers, which hold no conversion code, and they
//! were not read either), no FFmpeg source was read online, and no torchcodec
//! source was read for this module. The matrices are the public ITU-R BT.601,
//! BT.709 and BT.2020, FCC and SMPTE 240M definitions with the limited-range
//! scaling of H.273. The shape of the paths (a luma table plus integer chroma
//! offsets; 13-bit coefficients through a 16-bit multiply-high; a direct
//! 13-bit sum; 8-bit quantisation of subsampled high-bit-depth input) was
//! written from a general recollection of how libswscale is organised, which
//! the author cannot rule out came from having seen its source at some
//! earlier time, and every constant, rounding step and edge rule was then fixed by
//! black-box comparison with the `ffmpeg` binary. The experiments, in order:
//!
//! 1. 8-bit 4:2:0 (lane V): the table and NEON converters against all
//!    `(Y, Cb, Cr)` triples of BT.601, BT.709 and BT.2020, both ranges.
//! 2. 8-bit 4:2:2 against the 4:2:0 models on random planes at 9 widths: equal,
//!    with the chroma row of the luma row. Which of the two arithmetics runs, at
//!    sizes around 16, 32 and 64 with odd and even heights, is the NEON one iff
//!    the width is a multiple of 16 **and the height is even** (an odd-height 4:2:2
//!    picture takes the table converter whatever its width: [`Flavor::uses_neon`]).
//! 3. 8-bit 4:4:4: random planes matched neither 4:2:0 model, and a full cube of
//!    all 2^24 triples was independent of the scaler flags (bilinear, bicubic,
//!    neighbor, area, lanczos, accurate_rnd, full_chroma_int; not `fast_bilinear`)
//!    and of `-cpuflags 0`. Red depends on `(Y, Cr)` only and blue on `(Y, Cb)`
//!    only, linearly; a search over fixed-point forms found
//!    `(Y' * c13_y + Cr' * c13_rv + 4096) >> 13` with 0 differing red values, `c13`
//!    the 13-bit rounding of the 16.16 coefficient. Blue then differed for sums of
//!    512 or more (they read 0): the sum wraps as a 32-bit integer at 22
//!    fractional bits. With that, 0 differing bytes in all 2^24 triples of five
//!    matrices and both ranges.
//! 4. FCC and SMPTE 240M (their `Kr`, `Kb` from the public definitions) in the
//!    4:4:4 cube and in the 4:2:0 cube: exact on the table converter and on the
//!    NEON one once the two negative green coefficients are rounded as signed
//!    values (half up) and not as the negation of the rounded magnitude (lane V's
//!    full-range mismatch, 81,106 bytes of 2^24 for SMPTE 240M and 97,789 for FCC,
//!    was this tie rule).
//! 5. 10-bit 4:4:4: the 8-bit form at ten bits (`(Y' * c + Cr' * c + 2^14) >> 15`)
//!    gave 0 differing red values over the (Y, Cr) plane, and 2,064,000 sampled
//!    triples per matrix and range (edges of every range plus random) match in all
//!    three channels.
//! 6. 10-bit 4:2:2 (uniform pictures of 45,875 triples including every range
//!    edge): the 8-bit table converter on `(v + 2) >> 2` with luma not clipped
//!    (1022 and 1023 give 256) and chroma clipped to 255; any other rounding or
//!    clipping differs.
//! 7. 10-bit 4:2:0: the same, preceded by a vertical chroma filter. The filter's
//!    12-bit weights (sum 4096) were measured: impulse responses by slope, then
//!    linear-programming bounds on each weight from 4,096 random chroma samples per
//!    output row, then integer candidates held to full random pictures. `bilinear`:
//!    3072 and 1024 (rows `2j`: `j`, `j-1`; rows `2j+1`: `j`, `j+1`), clamped at the
//!    edges. `bicubic` (the CLI's default and what flags 0 give): interior
//!    `-115 985 3572 -346` (even rows, chroma `j-2..=j+1`) and `-346 3572 985 -115`
//!    (odd rows, `j-1..=j+2`), clamped at the bottom, with three measured top rows.
//!    Exact for even heights from 14 (bicubic) or 8 (bilinear) up, tested at a
//!    stride of heights to 1080; below that the scaler's filter is a different
//!    one (it shrinks with the source), which is refused by name here.
//! 8. Odd sizes: 4:2:2 with an odd height converts as with an even one. Odd
//!    height 4:2:0 resamples chroma by a ratio that is not 2 (the weights differ per
//!    row, and per flag); odd width leaves the **last output column unwritten**
//!    for 8-bit 4:2:0 and 4:2:2 (a destination buffer pre-filled with 0xAB keeps it),
//!    so the reference's bytes there are whatever its buffer held. 4:4:4 and the 10-bit
//!    formats write every column. H.264 and HEVC cannot code those 4:2:0 and 4:2:2
//!    shapes (their conformance window crops in units of the chroma sample), so they
//!    do not arise; this module refuses them by name.
//! 9. Rotation: the CLI's autorotation turns an 8-bit or 10-bit 4:2:0 picture's
//!    **planes** before the colour conversion (so the converter sees the upright
//!    width, and the 10-bit vertical chroma filter the upright rows) and a 4:2:2 or
//!    4:4:4 picture's **RGB** after it. On a 48x40 rotated-90 8-bit 4:2:0 fixture the
//!    two orders differ in 11,185 of 11,520 bytes. `media::video` follows the CLI.
//!    Which order torchcodec has is not known here.
//!
//! The scaler flags torchcodec uses are not known here (its source was not read
//! and it is not installed). They change one class: 10-bit 4:2:0. The 8-bit 4:2:0 and
//! 4:2:2 and all 4:4:4 results are the same under `bilinear`, `bicubic`, `neighbor`,
//! `area` and the others tried. [`ChromaFilter`] chooses; the default is
//! [`ChromaFilter::Bilinear`], the recollection of torchcodec's own choice
//! (`SWS_BILINEAR`), and [`ChromaFilter::Bicubic`] is what the `ffmpeg` CLI and
//! `sws_getContext` with flags 0 give.

use std::fmt;

/// Colour matrices whose libswscale coefficients are held here. `ycgco` and
/// BT.2020 constant luminance are not converted by this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    /// ITU-R BT.709 (H.273 `matrix_coefficients` 1).
    Bt709,
    /// ITU-R BT.601 / BT.470 BG / SMPTE 170M (6, 5), and what libswscale uses
    /// for a stream that does not say (2).
    Bt601,
    /// ITU-R BT.2020 non-constant luminance (9).
    Bt2020Nc,
    /// FCC (4): `Kr = 0.30`, `Kb = 0.11`.
    Fcc,
    /// SMPTE 240M (7): `Kr = 0.212`, `Kb = 0.087`.
    Smpte240m,
}

impl Matrix {
    /// The matrix libswscale uses for an H.273 `matrix_coefficients` code, or
    /// `None` for one this module does not hold. Code 2 (unspecified) is BT.601,
    /// as in libswscale.
    pub fn from_code(code: u8) -> Option<Matrix> {
        match code {
            1 => Some(Matrix::Bt709),
            2 | 5 | 6 => Some(Matrix::Bt601),
            4 => Some(Matrix::Fcc),
            7 => Some(Matrix::Smpte240m),
            9 => Some(Matrix::Bt2020Nc),
            _ => None,
        }
    }

    /// `[crv, cbu, cgu, cgv]` in 16.16, for chroma that spans the full
    /// 0..=255 (the values libswscale hands to its converters before range
    /// scaling): `2(1-Kr)`, `2(1-Kb)`, `2Kb(1-Kb)/Kg`, `2Kr(1-Kr)/Kg`, each times
    /// `65536 * 255 / 224` and rounded.
    pub fn coefficients(self) -> [i64; 4] {
        match self {
            Matrix::Bt709 => [117_489, 138_438, 13_975, 34_925],
            Matrix::Bt601 => [104_597, 132_201, 25_675, 53_279],
            Matrix::Bt2020Nc => [110_014, 140_363, 12_277, 42_626],
            Matrix::Fcc => [104_448, 132_798, 24_759, 53_109],
            Matrix::Smpte240m => [117_579, 136_230, 16_907, 35_559],
        }
    }
}

/// Whether Y spans 16..=235 (and chroma 16..=240) or 0..=255 (scaled by
/// `2^(depth-8)` above eight bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Range {
    Limited,
    Full,
}

/// Which libswscale converter runs, as a function of the frame width, for
/// 8-bit 4:2:0 and 4:2:2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    /// The aarch64 build of libswscale: the NEON converter for widths that are
    /// a multiple of 16, the table converter for the rest.
    Aarch64,
    /// The table converter for every width (what a CPU without the assembly
    /// version computes).
    Portable,
}

impl Flavor {
    /// Whether this build of libswscale runs its NEON converter on a picture of
    /// this size: a width that is a multiple of 16 and an even height (an odd-height
    /// 4:2:2 picture, whatever its width, takes the table converter).
    pub fn uses_neon(self, width: usize, height: usize) -> bool {
        self == Flavor::Aarch64 && width % 16 == 0 && height % 2 == 0
    }
}

/// The vertical chroma filter of the scaler for 10-bit 4:2:0: libswscale
/// resamples the chroma rows to luma rows, and the scaler flag decides how.
/// (The other formats do not depend on it.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChromaFilter {
    /// `SWS_BILINEAR` (the default here: the flag torchcodec is recalled to set).
    #[default]
    Bilinear,
    /// `SWS_BICUBIC`, the `ffmpeg` CLI's default and what unset flags give.
    Bicubic,
}

/// How chroma is subsampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chroma {
    /// Half the width and half the height.
    C420,
    /// Half the width, full height.
    C422,
    /// Full size.
    C444,
}

/// A sample format the converter accepts: bit depth and chroma layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub chroma: Chroma,
    /// 8 or 10.
    pub depth: u8,
}

/// How many threads convert rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Threads {
    /// One thread (the reference arrangement every other setting must equal).
    Single,
    /// As many as the machine has, up to eight, and only for pictures big enough
    /// to pay for them.
    #[default]
    Auto,
    /// This many (at least 1, at most one per row).
    Count(usize),
}

/// The settings of a conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub matrix: Matrix,
    pub range: Range,
    pub flavor: Flavor,
    pub filter: ChromaFilter,
    pub threads: Threads,
}

impl Params {
    /// `matrix` and `range` with libswscale-on-aarch64's converters, the
    /// default chroma filter and automatic threading.
    pub fn new(matrix: Matrix, range: Range) -> Params {
        Params {
            matrix,
            range,
            flavor: Flavor::Aarch64,
            filter: ChromaFilter::default(),
            threads: Threads::default(),
        }
    }
}

/// Why a conversion was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum YuvError {
    /// The width or height is zero, or odd where the chroma layout cannot be
    /// (4:2:0 and 4:2:2 need an even width, 4:2:0 an even height: libswscale
    /// leaves the last column of an odd-width 8-bit picture unwritten and
    /// resamples chroma unevenly for an odd height, which this module does not
    /// reproduce; H.264 and HEVC cannot code those shapes).
    Size { width: usize, height: usize },
    /// A 10-bit 4:2:0 picture shorter than the heights the vertical chroma filter
    /// was measured for (14 rows for bicubic, 8 for bilinear): below them the
    /// scaler builds another filter.
    TooShort {
        height: usize,
        filter: ChromaFilter,
        min: usize,
    },
    /// A bit depth other than 8 or 10.
    Depth { depth: u8 },
    /// A plane is shorter than its stride and height need.
    Plane {
        plane: &'static str,
        have: usize,
        need: usize,
    },
    /// A sample does not fit the bit depth.
    Sample { plane: &'static str, value: u32, depth: u8 },
}

impl fmt::Display for YuvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YuvError::Size { width, height } => write!(
                f,
                "a {width}x{height} picture cannot be converted as libswscale does: the sides must be non-zero, \
                 even where the chroma is subsampled (width for 4:2:0 and 4:2:2, height for 4:2:0)"
            ),
            YuvError::TooShort { height, filter, min } => write!(
                f,
                "a {height}-row 10-bit 4:2:0 picture is below the {min} rows the {filter:?} chroma filter is exact for"
            ),
            YuvError::Depth { depth } => write!(f, "{depth}-bit samples are not converted (8 and 10 are)"),
            YuvError::Plane { plane, have, need } => {
                write!(f, "the {plane} plane holds {have} samples, {need} are needed")
            }
            YuvError::Sample { plane, value, depth } => {
                write!(f, "a {plane} sample {value} does not fit {depth} bits")
            }
        }
    }
}

impl std::error::Error for YuvError {}

/// A sample type: `u8` for 8-bit planes, `u16` for the deeper ones (values
/// right-aligned, as FFmpeg's `yuv420p10le` holds them).
pub trait Sample: Copy + Send + Sync + 'static {
    /// Width of the type in bits.
    const BITS: u32;
    /// The value as an integer.
    fn get(self) -> u32;
}

impl Sample for u8 {
    const BITS: u32 = 8;
    #[inline(always)]
    fn get(self) -> u32 {
        u32::from(self)
    }
}

impl Sample for u16 {
    const BITS: u32 = 16;
    #[inline(always)]
    fn get(self) -> u32 {
        u32::from(self)
    }
}

/// A bi-planar picture: `y` is `height` rows of `width` samples, `uv` holds
/// interleaved Cb, Cr pairs, `chroma_height` rows of `2 * chroma_width` samples
/// (NV12, NV16, NV24, P010 and their kin once the samples are right-aligned);
/// each row starts `*_stride` samples after the last.
#[derive(Clone, Copy, Debug)]
pub struct Picture<'a, T: Sample> {
    pub width: usize,
    pub height: usize,
    pub layout: Layout,
    pub y: &'a [T],
    pub y_stride: usize,
    pub uv: &'a [T],
    pub uv_stride: usize,
}

/// An 8-bit bi-planar 4:2:0 picture (NV12): `y` is `height` rows of `width`
/// samples, `uv` is `height / 2` rows of `width / 2` interleaved Cb, Cr pairs;
/// each row starts `*_stride` bytes after the last.
#[derive(Clone, Copy, Debug)]
pub struct Nv12<'a> {
    pub width: usize,
    pub height: usize,
    pub y: &'a [u8],
    pub y_stride: usize,
    pub uv: &'a [u8],
    pub uv_stride: usize,
}

/// The 16.16 values every converter starts from, for a matrix and range: the
/// luma gain and the four chroma coefficients, scaled for the range.
struct Scaled {
    cy: i64,
    crv: i64,
    cbu: i64,
    cgu: i64,
    cgv: i64,
    /// Luma offset, in whole 8-bit levels.
    luma_offset: i64,
    full: bool,
}

/// C division: truncation toward zero.
fn cdiv(a: i64, b: i64) -> i64 {
    a / b
}

impl Scaled {
    fn new(matrix: Matrix, range: Range) -> Scaled {
        let [crv, cbu, cgu, cgv] = matrix.coefficients();
        let (crv, cbu, cgu, cgv) = (crv, cbu, -cgu, -cgv);
        match range {
            Range::Limited => Scaled {
                cy: cdiv(65536 * 255, 219),
                crv,
                cbu,
                cgu,
                cgv,
                luma_offset: 16,
                full: false,
            },
            Range::Full => Scaled {
                cy: 65536,
                crv: cdiv(crv * 224, 255),
                cbu: cdiv(cbu * 224, 255),
                cgu: cdiv(cgu * 224, 255),
                cgv: cdiv(cgv * 224, 255),
                luma_offset: 0,
                full: true,
            },
        }
    }
}

/// `roundToInt16` of an already-shifted 16.16 value: `(f + 0x8000) >> 16`,
/// saturated to the int16 range. The shift is arithmetic, so a negative value
/// rounds half up (toward +infinity), not away from zero.
fn round_to_i16(f: i64) -> i64 {
    ((f + (1 << 15)) >> 16).clamp(-0x7FFF, 0x7FFF)
}

/// `sqdmulh` on int16 operands: `(2ab) >> 16`.
fn sqdmulh(a: i64, b: i64) -> i64 {
    (2 * a * b) >> 16
}

fn clip_u8(v: i64) -> u8 {
    v.clamp(0, 255) as u8
}

/// The NEON converter: per `Y`, `Cb`, `Cr` terms in half levels, summed and
/// rounded back with `(x + 1) >> 1`.
struct Neon {
    /// `sqdmulh((Y - luma_offset) << 3, y_coeff)` per `Y`.
    luma: [i16; 256],
    vr: [i16; 256],
    ub: [i16; 256],
    ug: [i16; 256],
    vg: [i16; 256],
}

impl Neon {
    fn new(s: &Scaled) -> Neon {
        let y13 = round_to_i16(s.cy << 13);
        let vr13 = round_to_i16(s.crv << 13);
        let ub13 = round_to_i16(s.cbu << 13);
        // The green coefficients are negative; they are rounded as signed values.
        let ug13 = round_to_i16(s.cgu << 13);
        let vg13 = round_to_i16(s.cgv << 13);
        let mut n = Neon {
            luma: [0; 256],
            vr: [0; 256],
            ub: [0; 256],
            ug: [0; 256],
            vg: [0; 256],
        };
        for i in 0..256i64 {
            let c = (i - 128) << 3;
            n.luma[i as usize] = sqdmulh((i - s.luma_offset) << 3, y13) as i16;
            n.vr[i as usize] = sqdmulh(c, vr13) as i16;
            n.ub[i as usize] = sqdmulh(c, ub13) as i16;
            n.ug[i as usize] = sqdmulh(c, ug13) as i16;
            n.vg[i as usize] = sqdmulh(c, vg13) as i16;
        }
        n
    }

    #[inline(always)]
    fn pixel(&self, y: usize, u: usize, v: usize) -> [u8; 3] {
        let s = i64::from(self.luma[y]);
        let r = s + i64::from(self.vr[v]);
        let g = s + i64::from(self.ug[u]) + i64::from(self.vg[v]);
        let b = s + i64::from(self.ub[u]);
        [clip_u8((r + 1) >> 1), clip_u8((g + 1) >> 1), clip_u8((b + 1) >> 1)]
    }
}

/// The table converter: `out = T[Y + e + dc]`, with `T[k]` the clipped
/// `(k * cy - K + 0x8000) >> 16` and `dc` an integer offset per chroma value.
/// `Y` may reach 256 and a chroma value 255 (the 10-bit paths round to 8 bits
/// without clipping luma).
struct Table {
    /// `T[k - base]` for every `k` a pixel can index.
    t: Vec<u8>,
    base: i64,
    e: i64,
    rv: [i32; 256],
    bu: [i32; 256],
    gu: [i32; 256],
    gv: [i32; 256],
}

impl Table {
    fn new(s: &Scaled) -> Table {
        let inc = |c: i64| cdiv((c << 16) + 0x8000, s.cy.max(1));
        let (ir, ib, igu, igv) = (inc(s.crv), inc(s.cbu), inc(s.cgu), inc(s.cgv));
        let (e, k): (i64, i64) = if s.full { (384, 384 << 16) } else { (326, 400 << 16) };
        let offset = |i: i64, c: i64| ((c * i) >> 16) - (i >> 9);
        let mut t = Table {
            t: Vec::new(),
            base: 0,
            e,
            rv: [0; 256],
            bu: [0; 256],
            gu: [0; 256],
            gv: [0; 256],
        };
        for c in 0..256i64 {
            t.rv[c as usize] = offset(ir, c) as i32;
            t.bu[c as usize] = offset(ib, c) as i32;
            t.gu[c as usize] = offset(igu, c) as i32;
            t.gv[c as usize] = offset(igv, c) as i32;
        }
        let spread = |a: &[i32; 256]| {
            (
                i64::from(*a.iter().min().unwrap_or(&0)),
                i64::from(*a.iter().max().unwrap_or(&0)),
            )
        };
        let (rl, rh) = spread(&t.rv);
        let (bl, bh) = spread(&t.bu);
        let (ul, uh) = spread(&t.gu);
        let (vl, vh) = spread(&t.gv);
        let lo = e + rl.min(bl).min(ul + vl);
        let hi = e + 256 + rh.max(bh).max(uh + vh);
        t.base = lo;
        t.t = (lo..=hi).map(|i| clip_u8(((i * s.cy) - k + 0x8000) >> 16)).collect();
        t
    }

    #[inline(always)]
    fn at(&self, y: usize, d: i32) -> u8 {
        self.t[(y as i64 + self.e + i64::from(d) - self.base) as usize]
    }

    #[inline(always)]
    fn pixel(&self, y: usize, u: usize, v: usize) -> [u8; 3] {
        [
            self.at(y, self.rv[v]),
            self.at(y, self.gu[u] + self.gv[v]),
            self.at(y, self.bu[u]),
        ]
    }
}

/// The direct converter of 4:4:4 input: 13-bit coefficients, products and sums
/// in 32-bit integers with 22 fractional bits (so they wrap), result `>> 22`
/// clipped to a byte.
struct Direct {
    cy: i32,
    crv: i32,
    cbu: i32,
    cgu: i32,
    cgv: i32,
    /// Luma offset and chroma midpoint at the picture's depth.
    luma_offset: i32,
    mid: i32,
    /// `2^(22 - 13 - (depth - 8))`... the factor taking a sample-scale product to 22 fractional bits.
    scale: i32,
}

impl Direct {
    fn new(s: &Scaled, depth: u8) -> Direct {
        let up = i64::from(depth - 8);
        Direct {
            cy: round_to_i16(s.cy << 13) as i32,
            crv: round_to_i16(s.crv << 13) as i32,
            cbu: round_to_i16(s.cbu << 13) as i32,
            cgu: round_to_i16(s.cgu << 13) as i32,
            cgv: round_to_i16(s.cgv << 13) as i32,
            luma_offset: (s.luma_offset << up) as i32,
            mid: 128 << up,
            scale: 1 << (9 - up),
        }
    }

    #[inline(always)]
    fn pixel(&self, y: u32, u: u32, v: u32) -> [u8; 3] {
        let yv = (y as i32 - self.luma_offset)
            .wrapping_mul(self.cy)
            .wrapping_mul(self.scale)
            .wrapping_add(1 << 21);
        let (u, v) = (u as i32 - self.mid, v as i32 - self.mid);
        let r = yv.wrapping_add(v.wrapping_mul(self.crv).wrapping_mul(self.scale));
        let g = yv
            .wrapping_add(u.wrapping_mul(self.cgu).wrapping_mul(self.scale))
            .wrapping_add(v.wrapping_mul(self.cgv).wrapping_mul(self.scale));
        let b = yv.wrapping_add(u.wrapping_mul(self.cbu).wrapping_mul(self.scale));
        [clip_i32(r >> 22), clip_i32(g >> 22), clip_i32(b >> 22)]
    }
}

#[inline(always)]
fn clip_i32(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// The vertical chroma weights (12 bits, summing to 4096) of output row `r` of
/// a 10-bit 4:2:0 picture whose chroma has `ch` rows: `(chroma row, weight)`
/// pairs, rows already clamped to the picture.
fn chroma_taps(filter: ChromaFilter, ch: usize, r: usize) -> ([(usize, i32); 4], usize) {
    if ch == 1 {
        // A picture of two rows has one chroma row, whatever the filter.
        return ([(0, 4096), (0, 0), (0, 0), (0, 0)], 1);
    }
    let last = ch - 1;
    let clamp = |j: isize| j.clamp(0, last as isize) as usize;
    let j = (r / 2) as isize;
    match filter {
        ChromaFilter::Bilinear => {
            let other = if r % 2 == 0 { j - 1 } else { j + 1 };
            ([(clamp(j), 3072), (clamp(other), 1024), (0, 0), (0, 0)], 2)
        }
        ChromaFilter::Bicubic => match r {
            0 => ([(0, 4432), (1, -336), (0, 0), (0, 0)], 2),
            1 => ([(0, 3226), (1, 985), (2, -115), (0, 0)], 3),
            2 => ([(0, 959), (1, 3473), (2, -336), (0, 0)], 3),
            _ if r % 2 == 0 => (
                [
                    (clamp(j - 2), -115),
                    (clamp(j - 1), 985),
                    (clamp(j), 3572),
                    (clamp(j + 1), -346),
                ],
                4,
            ),
            _ => (
                [
                    (clamp(j - 1), -346),
                    (clamp(j), 3572),
                    (clamp(j + 1), 985),
                    (clamp(j + 2), -115),
                ],
                4,
            ),
        },
    }
}

/// The least height of a 10-bit 4:2:0 picture the vertical chroma filter was
/// measured for (below it the scaler builds a different filter); a picture of a
/// single chroma row is the same under every filter.
fn min_height(filter: ChromaFilter) -> usize {
    match filter {
        ChromaFilter::Bilinear => 8,
        ChromaFilter::Bicubic => 14,
    }
}

/// Whether a picture of this layout and size is one [`to_rgb24`] converts as
/// libswscale does.
///
/// # Errors
/// [`YuvError::Depth`], [`YuvError::Size`] or [`YuvError::TooShort`].
pub fn check_shape(layout: Layout, width: usize, height: usize, filter: ChromaFilter) -> Result<(), YuvError> {
    let Layout { chroma, depth } = layout;
    if depth != 8 && depth != 10 {
        return Err(YuvError::Depth { depth });
    }
    let size_error = YuvError::Size { width, height };
    if width == 0 || height == 0 {
        return Err(size_error);
    }
    if chroma != Chroma::C444 && width % 2 != 0 {
        return Err(size_error);
    }
    if chroma == Chroma::C420 && height % 2 != 0 {
        return Err(size_error);
    }
    if chroma == Chroma::C420 && depth > 8 && height > 2 && height < min_height(filter) {
        return Err(YuvError::TooShort {
            height,
            filter,
            min: min_height(filter),
        });
    }
    Ok(())
}

fn validate<T: Sample>(p: &Picture<'_, T>, filter: ChromaFilter) -> Result<(), YuvError> {
    check_shape(p.layout, p.width, p.height, filter)?;
    let Layout { chroma, depth } = p.layout;
    let need = |stride: usize, rows: usize, row_samples: usize| -> usize {
        if stride < row_samples || rows == 0 {
            usize::MAX
        } else {
            (rows - 1).saturating_mul(stride).saturating_add(row_samples)
        }
    };
    let (cw, ch) = chroma_size(p.width, p.height, chroma);
    let need_y = need(p.y_stride, p.height, p.width);
    if p.y.len() < need_y {
        return Err(YuvError::Plane {
            plane: "luma",
            have: p.y.len(),
            need: need_y,
        });
    }
    let need_uv = need(p.uv_stride, ch, 2 * cw);
    if p.uv.len() < need_uv {
        return Err(YuvError::Plane {
            plane: "chroma",
            have: p.uv.len(),
            need: need_uv,
        });
    }
    if T::BITS > u32::from(depth) {
        let max = (1u32 << depth) - 1;
        let worst = |plane: &[T], stride: usize, rows: usize, row_samples: usize| {
            (0..rows)
                .flat_map(|r| plane[r * stride..r * stride + row_samples].iter())
                .map(|s| s.get())
                .max()
                .unwrap_or(0)
        };
        for (name, value) in [
            ("luma", worst(p.y, p.y_stride, p.height, p.width)),
            ("chroma", worst(p.uv, p.uv_stride, ch, 2 * cw)),
        ] {
            if value > max {
                return Err(YuvError::Sample {
                    plane: name,
                    value,
                    depth,
                });
            }
        }
    }
    Ok(())
}

/// Width and height of the chroma planes.
fn chroma_size(width: usize, height: usize, chroma: Chroma) -> (usize, usize) {
    match chroma {
        Chroma::C420 => (width / 2, height / 2),
        Chroma::C422 => (width / 2, height),
        Chroma::C444 => (width, height),
    }
}

/// Threads to use for `rows` rows of `width` pixels.
fn worker_count(threads: Threads, width: usize, rows: usize) -> usize {
    let n = match threads {
        Threads::Single => 1,
        Threads::Count(n) => n.max(1),
        Threads::Auto => {
            if width.saturating_mul(rows) < (1 << 16) {
                1
            } else {
                std::thread::available_parallelism().map_or(1, |n| n.get()).min(8)
            }
        }
    };
    n.min(rows).max(1)
}

/// Run `fill(first_row, band)` over horizontal bands of `out` (`row_bytes` per
/// row), on `workers` threads. The bands are disjoint, so the bytes do not depend
/// on the count.
fn run_bands(out: &mut [u8], row_bytes: usize, workers: usize, fill: &(dyn Fn(usize, &mut [u8]) + Sync)) {
    let rows = out.len() / row_bytes;
    if workers <= 1 {
        fill(0, out);
        return;
    }
    let per = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut bands = out.chunks_mut(per * row_bytes).enumerate();
        let first = bands.next();
        for (i, band) in bands {
            scope.spawn(move || fill(i * per, band));
        }
        if let Some((i, band)) = first {
            fill(i * per, band);
        }
    });
}

/// 8-bit 4:2:0 and 4:2:2: chroma replicated over two luma samples (and for 4:2:0 two rows).
fn fill_replicated<T: Sample>(
    p: &Picture<'_, T>,
    first: usize,
    band: &mut [u8],
    px: &(impl Fn(usize, usize, usize) -> [u8; 3] + Sync),
) {
    let w = p.width;
    let vertical = p.layout.chroma == Chroma::C420;
    for (k, dst) in band.chunks_exact_mut(w * 3).enumerate() {
        let row = first + k;
        let crow = if vertical { row / 2 } else { row };
        let luma = &p.y[row * p.y_stride..row * p.y_stride + w];
        let chroma = &p.uv[crow * p.uv_stride..crow * p.uv_stride + w];
        for ((ys, cs), d) in luma
            .chunks_exact(2)
            .zip(chroma.chunks_exact(2))
            .zip(dst.chunks_exact_mut(6))
        {
            let (u, v) = (cs[0].get() as usize, cs[1].get() as usize);
            d[..3].copy_from_slice(&px(ys[0].get() as usize, u, v));
            d[3..].copy_from_slice(&px(ys[1].get() as usize, u, v));
        }
    }
}

/// 4:4:4: one chroma pair per luma sample.
fn fill_direct<T: Sample>(p: &Picture<'_, T>, d: &Direct, first: usize, band: &mut [u8]) {
    let w = p.width;
    for (k, dst) in band.chunks_exact_mut(w * 3).enumerate() {
        let row = first + k;
        let luma = &p.y[row * p.y_stride..row * p.y_stride + w];
        let chroma = &p.uv[row * p.uv_stride..row * p.uv_stride + 2 * w];
        for ((ys, cs), px) in luma.iter().zip(chroma.chunks_exact(2)).zip(dst.chunks_exact_mut(3)) {
            px.copy_from_slice(&d.pixel(ys.get(), cs[0].get(), cs[1].get()));
        }
    }
}

/// `(v + 2) >> 2`, the rounding of ten bits to eight that libswscale's scaler does.
#[inline(always)]
fn round_to_8(v: u32) -> usize {
    ((v + 2) >> 2) as usize
}

/// 10-bit 4:2:2 (`vertical` is `None`) and 4:2:0 (`Some(filter)`): luma and
/// chroma rounded to eight bits, chroma rows filtered for 4:2:0, then the table
/// converter.
fn fill_rounded<T: Sample>(p: &Picture<'_, T>, t: &Table, filter: Option<ChromaFilter>, first: usize, band: &mut [u8]) {
    let w = p.width;
    let (cw, ch) = chroma_size(p.width, p.height, p.layout.chroma);
    let mut cb = vec![0usize; cw];
    let mut cr = vec![0usize; cw];
    for (k, dst) in band.chunks_exact_mut(w * 3).enumerate() {
        let row = first + k;
        match filter {
            None => {
                let chroma = &p.uv[row * p.uv_stride..row * p.uv_stride + 2 * cw];
                for (c, pair) in chroma.chunks_exact(2).enumerate() {
                    cb[c] = round_to_8(pair[0].get()).min(255);
                    cr[c] = round_to_8(pair[1].get()).min(255);
                }
            }
            Some(filter) => {
                let (taps, n) = chroma_taps(filter, ch, row);
                for c in 0..cw {
                    let (mut su, mut sv) = (0i32, 0i32);
                    for &(r, wt) in &taps[..n] {
                        let at = r * p.uv_stride + 2 * c;
                        su += wt * p.uv[at].get() as i32;
                        sv += wt * p.uv[at + 1].get() as i32;
                    }
                    // 15-bit intermediates (value << 5) times 12-bit weights, rounded to 8 bits.
                    cb[c] = ((su + 8192) >> 14).clamp(0, 255) as usize;
                    cr[c] = ((sv + 8192) >> 14).clamp(0, 255) as usize;
                }
            }
        }
        let luma = &p.y[row * p.y_stride..row * p.y_stride + w];
        for (c, (ys, d)) in luma.chunks_exact(2).zip(dst.chunks_exact_mut(6)).enumerate() {
            d[..3].copy_from_slice(&t.pixel(round_to_8(ys[0].get()), cb[c], cr[c]));
            d[3..].copy_from_slice(&t.pixel(round_to_8(ys[1].get()), cb[c], cr[c]));
        }
    }
}

/// Convert a bi-planar picture to tightly packed `R, G, B` bytes
/// (`height * width * 3`), as libswscale does for the same size, matrix and range.
///
/// # Errors
/// [`YuvError::Size`] for a side the layout cannot have (zero; odd where the
/// chroma is subsampled), [`YuvError::TooShort`] for a 10-bit 4:2:0 picture
/// below the measured heights, [`YuvError::Depth`] for a depth other than 8 or
/// 10, [`YuvError::Plane`] for a plane the strides do not fit in and
/// [`YuvError::Sample`] for a sample wider than the depth.
pub fn to_rgb24<T: Sample>(p: &Picture<'_, T>, params: &Params) -> Result<Vec<u8>, YuvError> {
    validate(p, params.filter)?;
    let s = Scaled::new(params.matrix, params.range);
    let (w, h) = (p.width, p.height);
    let mut out = vec![0u8; w * h * 3];
    let workers = worker_count(params.threads, w, h);
    let Layout { chroma, depth } = p.layout;
    match (chroma, depth) {
        (Chroma::C444, _) => {
            let d = Direct::new(&s, depth);
            run_bands(&mut out, w * 3, workers, &|first, band| fill_direct(p, &d, first, band));
        }
        (_, 8) if params.flavor.uses_neon(w, h) => {
            let n = Neon::new(&s);
            run_bands(&mut out, w * 3, workers, &|first, band| {
                fill_replicated(p, first, band, &|y, u, v| n.pixel(y, u, v));
            });
        }
        (_, 8) => {
            let t = Table::new(&s);
            run_bands(&mut out, w * 3, workers, &|first, band| {
                fill_replicated(p, first, band, &|y, u, v| t.pixel(y, u, v));
            });
        }
        (_, _) => {
            let t = Table::new(&s);
            let filter = (chroma == Chroma::C420).then_some(params.filter);
            run_bands(&mut out, w * 3, workers, &|first, band| {
                fill_rounded(p, &t, filter, first, band);
            });
        }
    }
    Ok(out)
}

/// Convert an 8-bit NV12 picture to tightly packed `R, G, B` bytes
/// (`height * width * 3`) on one thread: [`to_rgb24`] for 8-bit 4:2:0.
///
/// # Errors
/// [`YuvError::Size`] for a zero or odd side, [`YuvError::Plane`] for a plane
/// the strides do not fit in.
pub fn nv12_to_rgb24(p: &Nv12<'_>, matrix: Matrix, range: Range, flavor: Flavor) -> Result<Vec<u8>, YuvError> {
    to_rgb24(
        &Picture {
            width: p.width,
            height: p.height,
            layout: Layout {
                chroma: Chroma::C420,
                depth: 8,
            },
            y: p.y,
            y_stride: p.y_stride,
            uv: p.uv,
            uv_stride: p.uv_stride,
        },
        &Params {
            flavor,
            threads: Threads::Single,
            ..Params::new(matrix, range)
        },
    )
}
