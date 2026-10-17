//! A pure-Rust baseline, progressive and arithmetic-coded JPEG decoder whose RGB
//! output is bit-identical to the one `torchvision.io.decode_image(path,
//! mode=RGB)` returns, which is what sentence-transformers runs on an image path
//! (libjpeg-turbo 3.1.3, 8-bit, `JDCT_ISLOW`, fancy upsampling, the merged
//! upsampler never selected, no block smoothing switch touched).
//!
//! # Why this exists
//!
//! ImageIO's JPEG decoder is not libjpeg-turbo: its chroma upsampling and IDCT
//! round differently (max 43, mean 0.75 per channel on a 4:2:0 picture), which
//! cost JPEG embeddings cosine 0.9986. This decoder reproduces the reference's
//! integer arithmetic instead.
//!
//! # How it is written
//!
//! The structure follows libjpeg's own control flow (`jdmarker.c`, `jdinput.c`,
//! `jdhuff.c`, `jdphuff.c`, `jdcoefct.c`, `jddctmgr.c`/`jidctint.c`, `jdsample.c`,
//! `jdcolor.c`), because "bit-identical" has to hold on damaged files too:
//! libjpeg warns and carries on at a bad Huffman code, at a marker inside entropy
//! data (the rest of the restart segment decodes as zero coefficients), at
//! extraneous bytes before a marker, and at a restart marker that is not the one
//! expected, and the reference decodes such files. What it does *not* do is read
//! past the end of its buffer: torchvision's source manager raises "Image is
//! incomplete or truncated" there, so every truncation, a missing EOI included,
//! is an error here too ([`JpegError::Truncated`]).
//!
//! Every size is checked before it allocates: the pixel count against the
//! caller's bound, the component, sampling-factor, MCU-block and Huffman-table
//! counts against libjpeg's own limits, marker lengths against the bytes that
//! exist, and the number of progressive scans against [`MAX_SCANS`].
//!
//! # Notice
//!
//! This decoder is based in part on the work of the Independent JPEG Group: it
//! reproduces libjpeg's design and arithmetic (no C source is included). The IJG
//! License's conditions and text are in `THIRD_PARTY_NOTICES.md` at the crate
//! root.

mod arith;
mod color;
mod exif;
mod huff;
mod idct;
mod markers;
mod output;
mod progressive;
mod sequential;
mod smooth;

use std::fmt;

use huff::{Derived, HuffSpec};

/// Largest image dimension libjpeg accepts (`JPEG_MAX_DIMENSION`).
pub const MAX_DIMENSION: u32 = 65500;
/// Most progressive (or other multi-)scan passes decoded. libjpeg has no bound; a
/// scan costs a pass over every block, so a file that spends 14 bytes per scan
/// could otherwise cost minutes. Typical encoders write 10 to 30.
pub const MAX_SCANS: usize = 256;
/// Most bytes of working memory (coefficients, planes and the output) one decode
/// may ask for, checked before it allocates.
pub const MAX_WORKING_BYTES: u64 = 6 << 30;

pub(crate) const MAX_COMPONENTS: usize = 10;
pub(crate) const MAX_COMPS_IN_SCAN: usize = 4;
pub(crate) const D_MAX_BLOCKS_IN_MCU: usize = 10;
pub(crate) const MAX_SAMP_FACTOR: usize = 4;
pub(crate) const NUM_TBLS: usize = 4;

/// Why a JPEG was not decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JpegError {
    /// The data ended where the reference stops: torchvision's source manager
    /// fails ("Image is incomplete or truncated") the moment libjpeg asks for a
    /// byte past the end, which includes a stream with no EOI.
    Truncated,
    /// There is no SOI at the start.
    NotJpeg,
    /// A condition libjpeg treats as fatal (`ERREXIT`), named.
    Invalid(String),
    /// A kind of JPEG this decoder does not decode, named.
    Unsupported(String),
    /// A bound of this decoder (not of the format) was exceeded.
    Limit(String),
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JpegError::Truncated => write!(f, "the data ends inside the JPEG stream (incomplete or truncated)"),
            JpegError::NotJpeg => write!(f, "no JPEG SOI marker at the start"),
            JpegError::Invalid(s) | JpegError::Unsupported(s) | JpegError::Limit(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for JpegError {}

pub(crate) type R<T> = Result<T, JpegError>;

pub(crate) fn invalid<T>(msg: impl Into<String>) -> R<T> {
    Err(JpegError::Invalid(msg.into()))
}

/// The colour space libjpeg guesses for the stored components
/// (`default_decompress_parms`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JpegColorSpace {
    Grayscale,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
    /// Two components, or more than four.
    Unknown,
}

/// How the entropy-coded data was written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JpegCoding {
    /// Baseline or extended sequential, Huffman (SOF0, SOF1).
    Sequential,
    /// Progressive, Huffman (SOF2).
    Progressive,
    /// Sequential, arithmetic (SOF9).
    SequentialArithmetic,
    /// Progressive, arithmetic (SOF10).
    ProgressiveArithmetic,
}

/// What the file said about itself, for callers that report it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegInfo {
    /// Components stored in the file (1, 3 or 4).
    pub components: usize,
    pub color_space: JpegColorSpace,
    pub coding: JpegCoding,
    /// The scans decoded (1 for a sequential file with all components in one).
    pub scans: usize,
    /// `(h, v)` sampling factors of each component, in file order.
    pub sampling: Vec<(u8, u8)>,
    /// A JFIF APP0 marker was present.
    pub jfif: bool,
    /// The Adobe APP14 transform byte, if there was an Adobe marker.
    pub adobe_transform: Option<u8>,
    /// IFD0 `Orientation` (0x0112) of the first Exif APP1 segment, 1..=8.
    pub exif_orientation: Option<u32>,
}

/// A decoded picture: `height * width * 3` bytes of RGB, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub info: JpegInfo,
}

/// Decode `bytes` to RGB the way `torchvision.io.decode_image(mode=RGB)` does.
/// `max_pixels` bounds `width * height`, and is checked from the header before
/// anything is allocated.
///
/// # Errors
/// See [`JpegError`]; no input panics.
pub fn decode(bytes: &[u8], max_pixels: usize) -> R<JpegImage> {
    check_signature(bytes)?;
    let mut d = Decoder::new(bytes);
    d.read_header()?;
    d.setup(max_pixels)?;
    d.read_all_scans()?;
    d.finish()
}

/// torchvision's `decode_image` hands data to the JPEG decoder only when it
/// starts with `FF D8 FF`; anything else is "unsupported image file" before
/// libjpeg sees it. So it is here.
pub fn has_jpeg_signature(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0xD8, 0xFF])
}

fn check_signature(bytes: &[u8]) -> R<()> {
    if has_jpeg_signature(bytes) {
        Ok(())
    } else {
        Err(JpegError::NotJpeg)
    }
}

/// One component of the frame (libjpeg's `jpeg_component_info`).
#[derive(Clone, Debug, Default)]
pub(crate) struct Comp {
    pub id: u8,
    pub h: usize,
    pub v: usize,
    pub tq: usize,
    pub width_in_blocks: usize,
    pub height_in_blocks: usize,
    pub downsampled_width: usize,
    pub downsampled_height: usize,
    /// Blocks per row / column of the padded (whole MCU) grid.
    pub blocks_w: usize,
    pub blocks_h: usize,
    /// The quantization table latched when the component first appears in a scan.
    pub quant: Option<[u16; 64]>,
    pub dc_tbl: usize,
    pub ac_tbl: usize,
    pub mcu_width: usize,
    pub mcu_height: usize,
    pub mcu_blocks: usize,
}

/// libjpeg's progression record for one component (`coef_bits`): per
/// coefficient the `Al` of the last scan that touched it, -1 for never.
pub(crate) type CoefBits = [i32; 64];

pub(crate) struct Src<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl Src<'_> {
    #[inline]
    pub fn byte(&mut self) -> R<u8> {
        match self.data.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                Ok(b)
            }
            None => Err(JpegError::Truncated),
        }
    }

    #[inline]
    pub fn u16(&mut self) -> R<u32> {
        let a = u32::from(self.byte()?);
        let b = u32::from(self.byte()?);
        Ok((a << 8) | b)
    }

    /// `skip_input_data`: past the end the reference's source manager substitutes
    /// a lone EOI byte, after which the next read fails, so this is the error.
    pub fn skip(&mut self, n: usize) -> R<()> {
        if n > self.data.len() - self.pos {
            self.pos = self.data.len();
            return Err(JpegError::Truncated);
        }
        self.pos += n;
        Ok(())
    }
}

/// The byte source with the entropy decoder's bit buffer (`jdhuff.c`'s bitread
/// state): kept apart from the tables so a table can be borrowed while bits are
/// read.
pub(crate) struct Input<'a> {
    pub src: Src<'a>,
    /// The marker that ended the entropy-coded data (or that `next_marker`
    /// found), 0 for none.
    pub unread_marker: u8,
    pub acc: u64,
    pub bits_left: i32,
    /// The segment ran out of data: the rest of it decodes as zeros.
    pub insufficient: bool,
}

/// Where decoded blocks go.
pub(crate) enum Store {
    /// Single-scan sequential: each block is inverse-transformed as it is
    /// decoded into a sample plane per component.
    Planes(Vec<Vec<u8>>),
    /// Everything else: a coefficient array per component, transformed once all
    /// scans are in.
    Coefs(Vec<Vec<i16>>),
}

pub(crate) struct Decoder<'a> {
    pub inp: Input<'a>,

    // Marker state (jdmarker.c).
    pub saw_soi: bool,
    pub saw_sof: bool,
    pub saw_jfif: bool,
    pub saw_adobe: bool,
    pub adobe_transform: u8,
    pub restart_interval: u32,
    pub next_restart_num: u8,
    pub qtbl: [Option<[u16; 64]>; NUM_TBLS],
    pub dc_spec: [Option<HuffSpec>; NUM_TBLS],
    pub ac_spec: [Option<HuffSpec>; NUM_TBLS],
    pub arith_dc_l: [u8; 16],
    pub arith_dc_u: [u8; 16],
    pub arith_ac_k: [u8; 16],
    pub exif_orientation: Option<u32>,

    // Frame (SOF).
    pub coding: JpegCoding,
    pub progressive: bool,
    pub arithmetic: bool,
    pub lossless: bool,
    pub precision: u32,
    pub width: u32,
    pub height: u32,
    pub comps: Vec<Comp>,
    pub max_h: usize,
    pub max_v: usize,
    pub mcus_per_row_full: usize,
    pub mcu_rows_full: usize,
    pub has_multiple_scans: bool,
    pub coef_bits: Vec<CoefBits>,

    // Scan.
    /// libjpeg's `cur_comp_info`, by position in the scan header.
    pub cur_comp: [Option<usize>; MAX_COMPS_IN_SCAN],
    pub scan_comps: Vec<usize>,
    /// For each block of an MCU: its slot in `scan_comps` and its column and row
    /// offset inside the component's share of the MCU.
    pub mcu_layout: Vec<(usize, usize, usize)>,
    pub ss: usize,
    pub se: usize,
    pub ah: u32,
    pub al: u32,
    pub mcus_per_row: usize,
    pub mcu_rows_in_scan: usize,
    pub blocks_in_mcu: usize,
    pub mcu_membership: [usize; D_MAX_BLOCKS_IN_MCU],
    pub input_scan_number: usize,
    pub eoi_reached: bool,

    // Entropy state.
    pub last_dc: [i32; MAX_COMPS_IN_SCAN],
    pub restarts_to_go: u32,
    pub eobrun: u32,
    pub dc_derived: [Option<Derived>; NUM_TBLS],
    pub ac_derived: [Option<Derived>; NUM_TBLS],
    pub arith: arith::State,

    pub store: Option<Store>,
}

impl<'a> Decoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Decoder {
            inp: Input {
                src: Src { data: bytes, pos: 0 },
                unread_marker: 0,
                acc: 0,
                bits_left: 0,
                insufficient: false,
            },
            saw_soi: false,
            saw_sof: false,
            saw_jfif: false,
            saw_adobe: false,
            adobe_transform: 0,
            restart_interval: 0,
            next_restart_num: 0,
            qtbl: [None; NUM_TBLS],
            dc_spec: [None, None, None, None],
            ac_spec: [None, None, None, None],
            arith_dc_l: [0; 16],
            arith_dc_u: [1; 16],
            arith_ac_k: [5; 16],
            exif_orientation: None,
            coding: JpegCoding::Sequential,
            progressive: false,
            arithmetic: false,
            lossless: false,
            precision: 8,
            width: 0,
            height: 0,
            comps: Vec::new(),
            max_h: 1,
            max_v: 1,
            mcus_per_row_full: 0,
            mcu_rows_full: 0,
            has_multiple_scans: false,
            coef_bits: Vec::new(),
            cur_comp: [None; MAX_COMPS_IN_SCAN],
            scan_comps: Vec::new(),
            mcu_layout: Vec::new(),
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
            mcus_per_row: 0,
            mcu_rows_in_scan: 0,
            blocks_in_mcu: 0,
            mcu_membership: [0; D_MAX_BLOCKS_IN_MCU],
            input_scan_number: 0,
            eoi_reached: false,
            last_dc: [0; MAX_COMPS_IN_SCAN],
            restarts_to_go: 0,
            eobrun: 0,
            dc_derived: [None, None, None, None],
            ac_derived: [None, None, None, None],
            arith: arith::State::default(),
            store: None,
        }
    }

    /// `jpeg_read_header`: markers up to and including the first SOS, then
    /// `initial_setup`.
    pub fn read_header(&mut self) -> R<()> {
        match self.read_markers()? {
            markers::Reached::Sos => {}
            markers::Reached::Eoi => return invalid("no image in the JPEG stream (EOI before any scan)"),
        }
        self.initial_setup()
    }

    /// What `jpeg_start_decompress` decides before any entropy data is read:
    /// the colour conversion must exist, the image must fit the bounds and the
    /// working memory must be affordable.
    pub fn setup(&mut self, max_pixels: usize) -> R<()> {
        let px = u64::from(self.width) * u64::from(self.height);
        if px > max_pixels as u64 {
            return Err(JpegError::Limit(format!(
                "a {}x{} image exceeds {max_pixels} pixels",
                self.height, self.width
            )));
        }
        // jinit_huff_decoder: a sequential Huffman stream that has not defined
        // tables 0 and 1 by now gets the standard ones. (Progressive and
        // arithmetic streams get none.)
        if !self.progressive && !self.arithmetic {
            let std = huff::standard_tables();
            for t in 0..2 {
                if self.dc_spec[t].is_none() {
                    self.dc_spec[t] = Some(std[t].clone());
                }
                if self.ac_spec[t].is_none() {
                    self.ac_spec[t] = Some(std[2 + t].clone());
                }
            }
        }
        match self.guess_color_space() {
            JpegColorSpace::Unknown => {
                return invalid(format!(
                    "{} components: no conversion to RGB (libjpeg: unsupported color conversion request)",
                    self.comps.len()
                ));
            }
            JpegColorSpace::Rgb | JpegColorSpace::YCbCr | JpegColorSpace::Grayscale => {}
            JpegColorSpace::Cmyk | JpegColorSpace::Ycck => {}
        }
        // Planes (one byte per padded sample), the coefficient arrays of a
        // multi-scan file (two bytes per coefficient) and the 3-byte output.
        let mut bytes = px * 3;
        for c in &self.comps {
            let samples = (c.blocks_w as u64) * (c.blocks_h as u64) * 64;
            bytes += samples;
            if self.has_multiple_scans {
                bytes += samples * 2;
            }
        }
        if bytes > MAX_WORKING_BYTES {
            return Err(JpegError::Limit(format!(
                "decoding a {}x{} image needs {bytes} bytes of working memory, over {MAX_WORKING_BYTES}",
                self.height, self.width
            )));
        }
        Ok(())
    }

    /// The colour space `default_decompress_parms` guesses.
    pub fn guess_color_space(&self) -> JpegColorSpace {
        match self.comps.len() {
            1 => JpegColorSpace::Grayscale,
            3 => {
                if self.saw_jfif {
                    JpegColorSpace::YCbCr
                } else if self.saw_adobe {
                    match self.adobe_transform {
                        0 => JpegColorSpace::Rgb,
                        _ => JpegColorSpace::YCbCr,
                    }
                } else {
                    let (a, b, c) = (self.comps[0].id, self.comps[1].id, self.comps[2].id);
                    if (a, b, c) == (b'R', b'G', b'B') {
                        JpegColorSpace::Rgb
                    } else {
                        JpegColorSpace::YCbCr
                    }
                }
            }
            4 => {
                if self.saw_adobe {
                    match self.adobe_transform {
                        0 => JpegColorSpace::Cmyk,
                        _ => JpegColorSpace::Ycck,
                    }
                } else {
                    JpegColorSpace::Cmyk
                }
            }
            _ => JpegColorSpace::Unknown,
        }
    }

    /// Decode every scan up to the EOI.
    pub fn read_all_scans(&mut self) -> R<()> {
        self.allocate_store();
        loop {
            self.start_input_pass()?;
            self.decode_scan()?;
            // `finish_input_pass` then `consume_markers`: the next SOS or the EOI.
            match self.read_markers()? {
                markers::Reached::Eoi => {
                    self.eoi_reached = true;
                    return Ok(());
                }
                markers::Reached::Sos => {
                    if !self.has_multiple_scans {
                        return invalid(
                            "a second scan in a file whose first scan holds every component (JERR_EOI_EXPECTED)",
                        );
                    }
                    if self.input_scan_number > MAX_SCANS {
                        return Err(JpegError::Limit(format!(
                            "more than {MAX_SCANS} scans (libjpeg has no bound; this decoder does)"
                        )));
                    }
                }
            }
        }
    }

    fn allocate_store(&mut self) {
        self.store = Some(if self.has_multiple_scans {
            Store::Coefs(
                self.comps
                    .iter()
                    .map(|c| vec![0i16; c.blocks_w * c.blocks_h * 64])
                    .collect(),
            )
        } else {
            Store::Planes(
                self.comps
                    .iter()
                    .map(|c| vec![0u8; c.blocks_w * c.blocks_h * 64])
                    .collect(),
            )
        });
    }

    fn info(&self) -> JpegInfo {
        JpegInfo {
            components: self.comps.len(),
            color_space: self.guess_color_space(),
            coding: self.coding,
            scans: self.input_scan_number,
            sampling: self.comps.iter().map(|c| (c.h as u8, c.v as u8)).collect(),
            jfif: self.saw_jfif,
            adobe_transform: self.saw_adobe.then_some(self.adobe_transform),
            exif_orientation: self.exif_orientation,
        }
    }
}
