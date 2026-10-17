//! Marker parsing: `jdmarker.c` (`read_markers`, `get_sof`, `get_sos`, ...) and
//! the parts of `jdinput.c` that turn a frame header into component geometry.

use super::huff::HuffSpec;
use super::{
    invalid, Comp, Decoder, JpegCoding, JpegError, D_MAX_BLOCKS_IN_MCU, MAX_COMPONENTS, MAX_COMPS_IN_SCAN,
    MAX_DIMENSION, MAX_SAMP_FACTOR, NUM_TBLS, R,
};

/// What `read_markers` stopped at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reached {
    Sos,
    Eoi,
}

pub(crate) const M_SOF0: u8 = 0xC0;
pub(crate) const M_SOF1: u8 = 0xC1;
pub(crate) const M_SOF2: u8 = 0xC2;
pub(crate) const M_SOF3: u8 = 0xC3;
pub(crate) const M_DHT: u8 = 0xC4;
pub(crate) const M_SOF9: u8 = 0xC9;
pub(crate) const M_SOF10: u8 = 0xCA;
pub(crate) const M_DAC: u8 = 0xCC;
pub(crate) const M_RST0: u8 = 0xD0;
pub(crate) const M_RST7: u8 = 0xD7;
pub(crate) const M_SOI: u8 = 0xD8;
pub(crate) const M_EOI: u8 = 0xD9;
pub(crate) const M_SOS: u8 = 0xDA;
pub(crate) const M_DQT: u8 = 0xDB;
pub(crate) const M_DNL: u8 = 0xDC;
pub(crate) const M_DRI: u8 = 0xDD;
pub(crate) const M_APP0: u8 = 0xE0;
pub(crate) const M_APP1: u8 = 0xE1;
pub(crate) const M_APP14: u8 = 0xEE;
pub(crate) const M_APP15: u8 = 0xEF;
pub(crate) const M_COM: u8 = 0xFE;
pub(crate) const M_TEM: u8 = 0x01;

/// `jpeg_natural_order` with libjpeg's 16 extra entries, which send a run past
/// coefficient 63 (damaged data) to coefficient 63.
pub(crate) const NATURAL_ORDER: [u8; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
    47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
];

impl Decoder<'_> {
    /// `first_marker`: the stream must start with SOI.
    fn first_marker(&mut self) -> R<()> {
        let c = self.inp.src.byte()?;
        let c2 = self.inp.src.byte()?;
        if c != 0xFF || c2 != M_SOI {
            return Err(JpegError::NotJpeg);
        }
        self.inp.unread_marker = c2;
        Ok(())
    }

    /// `next_marker`: skip to the next marker, swallowing fill bytes and
    /// stuffed zeros and counting nothing (a warning in libjpeg).
    pub fn next_marker(&mut self) -> R<()> {
        loop {
            let mut c = self.inp.src.byte()?;
            while c != 0xFF {
                c = self.inp.src.byte()?;
            }
            loop {
                c = self.inp.src.byte()?;
                if c != 0xFF {
                    break;
                }
            }
            if c != 0 {
                self.inp.unread_marker = c;
                return Ok(());
            }
        }
    }

    /// `read_markers`: process markers until a scan header or the EOI.
    pub fn read_markers(&mut self) -> R<Reached> {
        loop {
            if self.inp.unread_marker == 0 {
                if self.saw_soi {
                    self.next_marker()?;
                } else {
                    self.first_marker()?;
                }
            }
            match self.inp.unread_marker {
                M_SOI => self.get_soi()?,
                M_SOF0 | M_SOF1 => self.get_sof(JpegCoding::Sequential, false)?,
                M_SOF2 => self.get_sof(JpegCoding::Progressive, false)?,
                M_SOF9 => self.get_sof(JpegCoding::SequentialArithmetic, false)?,
                M_SOF10 => self.get_sof(JpegCoding::ProgressiveArithmetic, false)?,
                M_SOF3 => self.get_sof(JpegCoding::Sequential, true)?,
                0xC5..=0xC8 | 0xCB | 0xCD..=0xCF => {
                    let m = self.inp.unread_marker;
                    return Err(JpegError::Unsupported(format!(
                        "SOF marker 0x{m:02X} ({}): not decoded by the reference either",
                        sof_name(m)
                    )));
                }
                M_SOS => {
                    self.get_sos()?;
                    self.inp.unread_marker = 0;
                    return Ok(Reached::Sos);
                }
                M_EOI => {
                    self.inp.unread_marker = 0;
                    return Ok(Reached::Eoi);
                }
                M_DAC => self.get_dac()?,
                M_DHT => self.get_dht()?,
                M_DQT => self.get_dqt()?,
                M_DRI => self.get_dri()?,
                M_APP0 | M_APP14 => self.get_interesting_appn()?,
                M_APP1 => self.get_app1()?,
                0xE2..=0xED | M_APP15 | M_COM | M_DNL => self.skip_variable()?,
                0xD0..=0xD7 | M_TEM => {}
                m => return invalid(format!("unknown marker 0x{m:02X} (JERR_UNKNOWN_MARKER)")),
            }
            self.inp.unread_marker = 0;
        }
    }

    fn get_soi(&mut self) -> R<()> {
        if self.saw_soi {
            return invalid("duplicate SOI (JERR_SOI_DUPLICATE)");
        }
        self.arith_dc_l = [0; 16];
        self.arith_dc_u = [1; 16];
        self.arith_ac_k = [5; 16];
        self.restart_interval = 0;
        self.saw_jfif = false;
        self.saw_adobe = false;
        self.adobe_transform = 0;
        self.saw_soi = true;
        Ok(())
    }

    fn get_sof(&mut self, coding: JpegCoding, lossless: bool) -> R<()> {
        self.coding = coding;
        self.progressive = matches!(coding, JpegCoding::Progressive | JpegCoding::ProgressiveArithmetic);
        self.arithmetic = matches!(
            coding,
            JpegCoding::SequentialArithmetic | JpegCoding::ProgressiveArithmetic
        );
        self.lossless = lossless;
        let length = self.inp.src.u16()? as i64;
        self.precision = u32::from(self.inp.src.byte()?);
        self.height = self.inp.src.u16()?;
        self.width = self.inp.src.u16()?;
        let n = usize::from(self.inp.src.byte()?);
        let length = length - 8;
        if self.saw_sof {
            return invalid("duplicate SOF (JERR_SOF_DUPLICATE)");
        }
        if self.height == 0 || self.width == 0 || n == 0 {
            return invalid("empty image: a zero height, width or component count (JERR_EMPTY_IMAGE)");
        }
        if length != (n * 3) as i64 {
            return invalid("SOF length does not match the component count (JERR_BAD_LENGTH)");
        }
        self.comps = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.inp.src.byte()?;
            let hv = self.inp.src.byte()?;
            let tq = self.inp.src.byte()?;
            self.comps.push(Comp {
                id,
                h: usize::from(hv >> 4),
                v: usize::from(hv & 15),
                tq: usize::from(tq),
                ..Comp::default()
            });
        }
        self.saw_sof = true;
        Ok(())
    }

    fn get_sos(&mut self) -> R<()> {
        if !self.saw_sof {
            return invalid("SOS before SOF (JERR_SOS_NO_SOF)");
        }
        let length = self.inp.src.u16()? as i64;
        let n = usize::from(self.inp.src.byte()?);
        if length != (n * 2 + 6) as i64 || n < 1 || n > MAX_COMPS_IN_SCAN {
            return invalid("SOS length does not match its component count (JERR_BAD_LENGTH)");
        }
        self.cur_comp = [None; MAX_COMPS_IN_SCAN];
        for i in 0..n {
            let cc = self.inp.src.byte()?;
            let c = self.inp.src.byte()?;
            let mut found = None;
            // libjpeg-turbo walks the first MAX_COMPS_IN_SCAN components and skips
            // an index that `cur_comp_info` already holds.
            for ci in 0..self.comps.len().min(MAX_COMPS_IN_SCAN) {
                if self.comps[ci].id == cc && self.cur_comp[ci].is_none() {
                    found = Some(ci);
                    break;
                }
            }
            let Some(ci) = found else {
                return invalid(format!(
                    "scan names component id {cc}, which the frame lacks (JERR_BAD_COMPONENT_ID)"
                ));
            };
            self.cur_comp[i] = Some(ci);
            self.comps[ci].dc_tbl = usize::from((c >> 4) & 15);
            self.comps[ci].ac_tbl = usize::from(c & 15);
        }
        self.scan_comps = self.cur_comp[..n].iter().map(|c| c.unwrap_or(0)).collect();
        self.ss = usize::from(self.inp.src.byte()?);
        self.se = usize::from(self.inp.src.byte()?);
        let a = self.inp.src.byte()?;
        self.ah = u32::from(a >> 4);
        self.al = u32::from(a & 15);
        self.next_restart_num = 0;
        self.input_scan_number += 1;
        Ok(())
    }

    fn get_dht(&mut self) -> R<()> {
        let mut length = self.inp.src.u16()? as i64 - 2;
        while length > 16 {
            let index = usize::from(self.inp.src.byte()?);
            let mut bits = [0u8; 17];
            let mut count = 0usize;
            for b in bits.iter_mut().skip(1) {
                *b = self.inp.src.byte()?;
                count += usize::from(*b);
            }
            length -= 17;
            if count > 256 || count as i64 > length {
                return invalid("Huffman table has more symbols than fit (JERR_BAD_HUFF_TABLE)");
            }
            let mut vals = [0u8; 256];
            for v in vals.iter_mut().take(count) {
                *v = self.inp.src.byte()?;
            }
            length -= count as i64;
            let (slot, which) = if index & 0x10 != 0 {
                (index - 0x10, 1)
            } else {
                (index, 0)
            };
            if slot >= NUM_TBLS {
                return invalid(format!("Huffman table index {index} (JERR_DHT_INDEX)"));
            }
            let spec = Some(HuffSpec { bits, vals });
            if which == 1 {
                self.ac_spec[slot] = spec;
            } else {
                self.dc_spec[slot] = spec;
            }
        }
        if length != 0 {
            return invalid("DHT length is wrong (JERR_BAD_LENGTH)");
        }
        Ok(())
    }

    fn get_dqt(&mut self) -> R<()> {
        let mut length = self.inp.src.u16()? as i64 - 2;
        while length > 0 {
            let n = self.inp.src.byte()?;
            let prec = n >> 4;
            let n = usize::from(n & 15);
            if n >= NUM_TBLS {
                return invalid(format!("quantization table index {n} (JERR_DQT_INDEX)"));
            }
            let mut q = [0u16; 64];
            for i in 0..64 {
                let v = if prec != 0 {
                    self.inp.src.u16()? as u16
                } else {
                    u16::from(self.inp.src.byte()?)
                };
                q[usize::from(NATURAL_ORDER[i])] = v;
            }
            self.qtbl[n] = Some(q);
            length -= 65;
            if prec != 0 {
                length -= 64;
            }
        }
        if length != 0 {
            return invalid("DQT length is wrong (JERR_BAD_LENGTH)");
        }
        Ok(())
    }

    fn get_dri(&mut self) -> R<()> {
        let length = self.inp.src.u16()?;
        if length != 4 {
            return invalid("DRI length is not 4 (JERR_BAD_LENGTH)");
        }
        self.restart_interval = self.inp.src.u16()?;
        Ok(())
    }

    fn get_dac(&mut self) -> R<()> {
        let mut length = self.inp.src.u16()? as i64 - 2;
        while length > 0 {
            let index = usize::from(self.inp.src.byte()?);
            let val = self.inp.src.byte()?;
            length -= 2;
            if index >= 32 {
                return invalid(format!("arithmetic conditioning index {index} (JERR_DAC_INDEX)"));
            }
            if index >= 16 {
                self.arith_ac_k[index - 16] = val;
            } else {
                self.arith_dc_l[index] = val & 15;
                self.arith_dc_u[index] = val >> 4;
                if self.arith_dc_l[index] > self.arith_dc_u[index] {
                    return invalid(format!("arithmetic conditioning value 0x{val:02X} (JERR_DAC_VALUE)"));
                }
            }
        }
        if length != 0 {
            return invalid("DAC length is wrong (JERR_BAD_LENGTH)");
        }
        Ok(())
    }

    /// `skip_variable`.
    fn skip_variable(&mut self) -> R<()> {
        let length = self.inp.src.u16()? as i64 - 2;
        if length > 0 {
            self.inp.src.skip(length as usize)?;
        }
        Ok(())
    }

    /// APP1: skipped like any other, but its bytes are read first because an Exif
    /// segment carries the orientation callers may want.
    fn get_app1(&mut self) -> R<()> {
        let length = self.inp.src.u16()? as i64 - 2;
        if length <= 0 {
            return Ok(());
        }
        let n = length as usize;
        let start = self.inp.src.pos;
        self.inp.src.skip(n)?;
        if self.exif_orientation.is_none() {
            self.exif_orientation = super::exif::orientation(&self.inp.src.data[start..start + n]);
        }
        Ok(())
    }

    /// `get_interesting_appn`: APP0 (JFIF) and APP14 (Adobe).
    fn get_interesting_appn(&mut self) -> R<()> {
        let marker = self.inp.unread_marker;
        let mut length = self.inp.src.u16()? as i64 - 2;
        let numtoread = if length >= 14 {
            14
        } else if length > 0 {
            length as usize
        } else {
            0
        };
        let mut b = [0u8; 14];
        for slot in b.iter_mut().take(numtoread) {
            *slot = self.inp.src.byte()?;
        }
        length -= numtoread as i64;
        if marker == M_APP0 {
            if numtoread >= 14 && &b[..5] == b"JFIF\0" {
                self.saw_jfif = true;
            }
        } else if numtoread >= 12 && &b[..5] == b"Adobe" {
            self.saw_adobe = true;
            self.adobe_transform = b[11];
        }
        if length > 0 {
            self.inp.src.skip(length as usize)?;
        }
        Ok(())
    }

    /// `initial_setup` (jdinput.c): validate the frame and compute each
    /// component's geometry.
    pub fn initial_setup(&mut self) -> R<()> {
        if self.width > MAX_DIMENSION || self.height > MAX_DIMENSION {
            return invalid(format!("image larger than {MAX_DIMENSION} pixels (JERR_IMAGE_TOO_BIG)"));
        }
        if self.lossless {
            return Err(JpegError::Unsupported(
                "lossless JPEG (SOF3): the reference decodes it, this decoder does not".into(),
            ));
        }
        match self.precision {
            8 => {}
            12 => {
                return Err(JpegError::Unsupported(
                    "12-bit JPEG: the reference's 8-bit decode path refuses it (JERR_BAD_PRECISION)".into(),
                ))
            }
            p => return invalid(format!("sample precision {p} (JERR_BAD_PRECISION)")),
        }
        let n = self.comps.len();
        if n > MAX_COMPONENTS {
            return invalid(format!(
                "{n} components, more than {MAX_COMPONENTS} (JERR_COMPONENT_COUNT)"
            ));
        }
        self.max_h = 1;
        self.max_v = 1;
        for c in &self.comps {
            if c.h == 0 || c.h > MAX_SAMP_FACTOR || c.v == 0 || c.v > MAX_SAMP_FACTOR {
                return invalid(format!(
                    "sampling factors {}x{} are outside 1..={MAX_SAMP_FACTOR} (JERR_BAD_SAMPLING)",
                    c.h, c.v
                ));
            }
            self.max_h = self.max_h.max(c.h);
            self.max_v = self.max_v.max(c.v);
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let (mh, mv) = (self.max_h, self.max_v);
        for c in &mut self.comps {
            c.width_in_blocks = (w * c.h).div_ceil(mh * 8);
            c.height_in_blocks = (h * c.v).div_ceil(mv * 8);
            c.downsampled_width = (w * c.h).div_ceil(mh);
            c.downsampled_height = (h * c.v).div_ceil(mv);
        }
        self.mcus_per_row_full = w.div_ceil(mh * 8);
        self.mcu_rows_full = h.div_ceil(mv * 8);
        for c in &mut self.comps {
            c.blocks_w = self.mcus_per_row_full * c.h;
            c.blocks_h = self.mcu_rows_full * c.v;
        }
        self.has_multiple_scans = self.scan_comps.len() < n || self.progressive;
        self.coef_bits = vec![[-1; 64]; n];
        Ok(())
    }

    /// `per_scan_setup`, `latch_quant_tables` and the entropy decoder's
    /// `start_pass`, in the order `start_input_pass` runs them.
    pub fn start_input_pass(&mut self) -> R<()> {
        self.per_scan_setup()?;
        // latch_quant_tables
        for &ci in &self.scan_comps.clone() {
            if self.comps[ci].quant.is_some() {
                continue;
            }
            let tq = self.comps[ci].tq;
            let Some(q) = self.qtbl.get(tq).copied().flatten() else {
                return invalid(format!("quantization table {tq} is not defined (JERR_NO_QUANT_TABLE)"));
            };
            self.comps[ci].quant = Some(q);
        }
        self.start_entropy_pass()
    }

    fn per_scan_setup(&mut self) -> R<()> {
        let n = self.scan_comps.len();
        if n == 1 {
            let ci = self.scan_comps[0];
            let c = &mut self.comps[ci];
            self.mcus_per_row = c.width_in_blocks;
            self.mcu_rows_in_scan = c.height_in_blocks;
            c.mcu_width = 1;
            c.mcu_height = 1;
            c.mcu_blocks = 1;
            self.blocks_in_mcu = 1;
            self.mcu_membership[0] = 0;
        } else {
            self.mcus_per_row = self.mcus_per_row_full;
            self.mcu_rows_in_scan = self.mcu_rows_full;
            self.blocks_in_mcu = 0;
            for k in 0..n {
                let ci = self.scan_comps[k];
                let c = &mut self.comps[ci];
                c.mcu_width = c.h;
                c.mcu_height = c.v;
                c.mcu_blocks = c.h * c.v;
                if self.blocks_in_mcu + c.mcu_blocks > D_MAX_BLOCKS_IN_MCU {
                    return invalid(format!(
                        "an MCU of more than {D_MAX_BLOCKS_IN_MCU} blocks (JERR_BAD_MCU_SIZE)"
                    ));
                }
                for _ in 0..c.mcu_blocks {
                    self.mcu_membership[self.blocks_in_mcu] = k;
                    self.blocks_in_mcu += 1;
                }
            }
        }
        Ok(())
    }

    /// `read_restart_marker` with `jpeg_resync_to_restart` behind it.
    pub fn read_restart_marker(&mut self) -> R<()> {
        if self.inp.unread_marker == 0 {
            self.next_marker()?;
        }
        if self.inp.unread_marker == M_RST0 + self.next_restart_num {
            self.inp.unread_marker = 0;
        } else {
            self.resync_to_restart()?;
        }
        self.next_restart_num = (self.next_restart_num + 1) & 7;
        Ok(())
    }

    fn resync_to_restart(&mut self) -> R<()> {
        let desired = i32::from(self.next_restart_num);
        let mut marker = i32::from(self.inp.unread_marker);
        loop {
            let action = if marker < i32::from(M_SOF0) {
                2
            } else if marker < i32::from(M_RST0) || marker > i32::from(M_RST7) {
                3
            } else {
                let m = marker - i32::from(M_RST0);
                if m == (desired + 1) & 7 || m == (desired + 2) & 7 {
                    3
                } else if m == (desired - 1) & 7 || m == (desired - 2) & 7 {
                    2
                } else {
                    1
                }
            };
            match action {
                1 => {
                    self.inp.unread_marker = 0;
                    return Ok(());
                }
                2 => {
                    self.next_marker()?;
                    marker = i32::from(self.inp.unread_marker);
                }
                _ => return Ok(()),
            }
        }
    }
}

fn sof_name(m: u8) -> &'static str {
    match m {
        0xC5 => "differential sequential Huffman",
        0xC6 => "differential progressive Huffman",
        0xC7 => "differential lossless Huffman",
        0xC8 => "reserved JPG",
        0xCB => "lossless arithmetic",
        0xCD => "differential sequential arithmetic",
        0xCE => "differential progressive arithmetic",
        _ => "differential lossless arithmetic",
    }
}
