//! YCbCr 4:2:0 to RGB, byte for byte what FFmpeg's libswscale computes for a
//! same-size `yuv420p` to `rgb24` conversion.
//!
//! This is the conversion the reference loader of EmbeddingGemma 2 runs.
//! `transformers` decodes a video file with torchcodec, and torchcodec turns
//! each decoded frame into RGB with libswscale (`SwScale.cpp`: the frame's own
//! colorspace and range, no scaling) or with the filter graph `format=rgb24`,
//! whose automatically inserted scaler is the same library. For a frame the
//! size of its output, libswscale does not filter: each luma sample takes the
//! chroma sample of its 2x2 block (no interpolation, whatever the scaler flags),
//! and the result is a function of `(Y, Cb, Cr)` only.
//!
//! # The two arithmetics
//!
//! libswscale has two implementations of that function, and which one runs
//! depends on the CPU and the width:
//!
//! * the **table** converter (`yuv2rgb.c`, `yuv2rgb_c_24_rgb`): a 1024-entry
//!   luma table, indexed by `Y` plus an integer chroma offset per `Cb`/`Cr`
//!   value. It is the C reference and the only one on CPUs without an assembly
//!   version;
//! * the **aarch64 NEON** converter (`aarch64/yuv2rgb_neon.S`): 16-bit
//!   `sqdmulh` arithmetic with 13-bit coefficients, which FFmpeg uses on arm64
//!   when the width is a multiple of 16. The two differ by up to 3 levels.
//!
//! [`Flavor::Aarch64`] picks per frame width as FFmpeg does on arm64 (and so
//! as the `ffmpeg` CLI and torchcodec do on Apple silicon); [`Flavor::Portable`]
//! is the table converter for every width.
//!
//! # Where the numbers come from
//!
//! Both converters are built from four 16.16 coefficients per colour matrix.
//! FFmpeg 9 derives them from the matrix's `Kr`/`Kb` as `2(1-Kr)`, `2(1-Kb)`,
//! `2Kb(1-Kb)/Kg` and `2Kr(1-Kr)/Kg`, times `65536 * 255 / 224` (chroma
//! excursion), rounded; [`Matrix::coefficients`] holds the integers. The NEON
//! coefficients are those `/ 8` rounded, the table converter's are the
//! coefficients divided by the luma gain. The arithmetic and the constants were
//! recovered from the `ffmpeg` binary (9.0.2, aarch64) and checked against it
//! for **every** `(Y, Cb, Cr)` triple of every matrix and range listed here, in
//! both modes (`tests/media_video_exact.rs`, `exhaustive_*`, needs `ffmpeg`).

use std::fmt;

/// Colour matrices whose libswscale coefficients are held here. Anything else
/// (`fcc`, `smpte240m`, `ycgco`) is not converted by this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    /// ITU-R BT.709 (H.273 `matrix_coefficients` 1).
    Bt709,
    /// ITU-R BT.601 / BT.470 BG / SMPTE 170M (6, 5), and what libswscale uses
    /// for a stream that does not say (2).
    Bt601,
    /// ITU-R BT.2020 non-constant luminance (9).
    Bt2020Nc,
}

impl Matrix {
    /// The matrix libswscale uses for an H.273 `matrix_coefficients` code, or
    /// `None` for one this module does not hold. Code 2 (unspecified) is BT.601,
    /// as in libswscale.
    pub fn from_code(code: u8) -> Option<Matrix> {
        match code {
            1 => Some(Matrix::Bt709),
            2 | 5 | 6 => Some(Matrix::Bt601),
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
        }
    }
}

/// Whether Y spans 16..=235 (and chroma 16..=240) or 0..=255.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Range {
    Limited,
    Full,
}

/// Which libswscale converter runs, as a function of the frame width.
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
    fn uses_neon(self, width: usize) -> bool {
        self == Flavor::Aarch64 && width % 16 == 0
    }
}

/// Why a conversion was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum YuvError {
    /// The width or height is zero or odd (a 4:2:0 picture from H.264 or HEVC
    /// has even sides; libswscale filters odd ones, which this module does not).
    Size { width: usize, height: usize },
    /// A plane is shorter than its stride and height need.
    Plane {
        plane: &'static str,
        have: usize,
        need: usize,
    },
}

impl fmt::Display for YuvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YuvError::Size { width, height } => {
                write!(f, "a {width}x{height} 4:2:0 frame must have even, non-zero sides")
            }
            YuvError::Plane { plane, have, need } => {
                write!(f, "the {plane} plane holds {have} bytes, {need} are needed")
            }
        }
    }
}

impl std::error::Error for YuvError {}

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

/// The 16.16 values both converters start from, for a matrix and range
/// (`ff_yuv2rgb_c_init_tables`): the luma gain and the four chroma
/// coefficients, scaled for the range.
struct Scaled {
    cy: i64,
    crv: i64,
    cbu: i64,
    cgu: i64,
    cgv: i64,
    /// Luma offset, in whole levels.
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
/// saturated to the int16 range.
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
        let ug13 = round_to_i16((-s.cgu) << 13);
        let vg13 = round_to_i16((-s.cgv) << 13);
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
            n.ug[i as usize] = sqdmulh(c, -ug13) as i16;
            n.vg[i as usize] = sqdmulh(c, -vg13) as i16;
        }
        n
    }

    #[inline]
    fn pixel(&self, y: u8, u: u8, v: u8) -> [u8; 3] {
        let s = i64::from(self.luma[usize::from(y)]);
        let r = s + i64::from(self.vr[usize::from(v)]);
        let g = s + i64::from(self.ug[usize::from(u)]) + i64::from(self.vg[usize::from(v)]);
        let b = s + i64::from(self.ub[usize::from(u)]);
        [clip_u8((r + 1) >> 1), clip_u8((g + 1) >> 1), clip_u8((b + 1) >> 1)]
    }
}

/// The table converter: `out = T[Y + e + dc]`, with `T[k]` the clipped
/// `(k * cy - K + 0x8000) >> 16` and `dc` an integer offset per chroma value.
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
        let hi = e + 255 + rh.max(bh).max(uh + vh);
        t.base = lo;
        t.t = (lo..=hi).map(|i| clip_u8(((i * s.cy) - k + 0x8000) >> 16)).collect();
        t
    }

    #[inline]
    fn at(&self, y: u8, d: i64) -> u8 {
        self.t[(i64::from(y) + self.e + d - self.base) as usize]
    }

    #[inline]
    fn pixel(&self, y: u8, u: u8, v: u8) -> [u8; 3] {
        let (u, v) = (usize::from(u), usize::from(v));
        [
            self.at(y, i64::from(self.rv[v])),
            self.at(y, i64::from(self.gu[u]) + i64::from(self.gv[v])),
            self.at(y, i64::from(self.bu[u])),
        ]
    }
}

fn check(p: &Nv12<'_>) -> Result<(), YuvError> {
    if p.width == 0 || p.height == 0 || p.width % 2 != 0 || p.height % 2 != 0 {
        return Err(YuvError::Size {
            width: p.width,
            height: p.height,
        });
    }
    let need = |stride: usize, rows: usize, row_bytes: usize| -> usize {
        if stride < row_bytes || rows == 0 {
            usize::MAX
        } else {
            (rows - 1).saturating_mul(stride).saturating_add(row_bytes)
        }
    };
    let need_y = need(p.y_stride, p.height, p.width);
    if p.y.len() < need_y {
        return Err(YuvError::Plane {
            plane: "luma",
            have: p.y.len(),
            need: need_y,
        });
    }
    let need_uv = need(p.uv_stride, p.height / 2, p.width);
    if p.uv.len() < need_uv {
        return Err(YuvError::Plane {
            plane: "chroma",
            have: p.uv.len(),
            need: need_uv,
        });
    }
    Ok(())
}

/// Convert an NV12 picture to tightly packed `R, G, B` bytes (`height * width * 3`).
///
/// # Errors
/// [`YuvError::Size`] for a zero or odd side, [`YuvError::Plane`] for a plane
/// the strides do not fit in.
pub fn nv12_to_rgb24(p: &Nv12<'_>, matrix: Matrix, range: Range, flavor: Flavor) -> Result<Vec<u8>, YuvError> {
    check(p)?;
    let s = Scaled::new(matrix, range);
    let mut out = vec![0u8; p.width * p.height * 3];
    if flavor.uses_neon(p.width) {
        let c = Neon::new(&s);
        fill(p, &mut out, |y, u, v| c.pixel(y, u, v));
    } else {
        let c = Table::new(&s);
        fill(p, &mut out, |y, u, v| c.pixel(y, u, v));
    }
    Ok(out)
}

fn fill(p: &Nv12<'_>, out: &mut [u8], px: impl Fn(u8, u8, u8) -> [u8; 3]) {
    let w = p.width;
    for row in 0..p.height {
        let luma = &p.y[row * p.y_stride..row * p.y_stride + w];
        let chroma = &p.uv[(row / 2) * p.uv_stride..(row / 2) * p.uv_stride + w];
        let dst = &mut out[row * w * 3..(row + 1) * w * 3];
        for ((ys, cs), d) in luma
            .chunks_exact(2)
            .zip(chroma.chunks_exact(2))
            .zip(dst.chunks_exact_mut(6))
        {
            let (u, v) = (cs[0], cs[1]);
            d[..3].copy_from_slice(&px(ys[0], u, v));
            d[3..].copy_from_slice(&px(ys[1], u, v));
        }
    }
}
