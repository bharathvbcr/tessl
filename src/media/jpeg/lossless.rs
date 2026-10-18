//! Lossless JPEG (SOF3, ITU-T T.81 Annex H) as libjpeg-turbo 3.x decodes it
//! (`jdlossls.c`, `jdlhuff.c`, `jddiffct.c`, `jdpred.c`, `jdscale.c`).
//!
//! A lossless scan codes, for every sample, the difference to a prediction made
//! from the samples to its left (`Ra`), above (`Rb`) and above-left (`Rc`):
//! the Huffman code names the difference's category (0 to 16) and the extra bits
//! give its value, exactly as a DC difference does in the DCT modes. What
//! follows is the reference's behaviour, established with the oracle on the
//! fixtures `lossless_*` and checked against the 3.1.3 source (`jddiffct.c`,
//! `jdlossls.c`, `jdlhuff.c`, `jdinput.c`; line numbers in the comments), not a
//! copy of it:
//!
//! * the unit of coding is one sample, so a "block" is a sample, an MCU holds
//!   `h * v` samples of each component, and the picture is `ceil(w / h_max)` MCUs
//!   wide;
//! * the first sample of a scan, and of every restart interval, is predicted
//!   from `2^(P - Pt - 1)` and the rest of its row from the left; the first sample
//!   of a later row is predicted from above; everything else by the scan's
//!   predictor selection value (`Ss`, 1 to 7). Restart intervals are whole MCU
//!   rows of the *scan* (`jddiffct.c:104-112`: `restart_interval % MCUs_per_row`,
//!   and a non-interleaved scan's MCU row is one sample row of its component);
//! * a scan is undifferenced one iMCU row at a time (`jddiffct.c:175-226`): an
//!   interleaved scan's MCU row is its iMCU row, a non-interleaved scan decodes
//!   `v_samp_factor` MCU rows first (restart markers are honoured between them)
//!   and undifferences them together afterwards, so a restart, or a row decoded
//!   from an exhausted segment, in the middle of an iMCU row makes the iMCU row's
//!   *first* row the one predicted from `2^(P - Pt - 1)`;
//! * the reconstruction is `(prediction + difference) mod 2^16`, kept in 16 bits
//!   as the next prediction's input; the sample stored is that value shifted left
//!   by the point transform `Al`, truncated to 8 bits;
//! * there is no DCT, so upsampling is plain replication (the "fancy" triangle
//!   filters belong to the DCT path).
//!
//! # Provenance
//!
//! The predictors are T.81 Table H.1; the rest is libjpeg's design reproduced
//! from its documented behaviour (see `THIRD_PARTY_NOTICES.md`); no C source is
//! included.

use super::huff::huff_extend;
use super::{invalid, Decoder, JpegError, Store, R};

/// The sample precisions the reference decodes (torchvision refuses above 8 bits: "Unsupported JPEG data precision").
pub(crate) fn precision_supported(p: u32) -> bool {
    (2..=8).contains(&p)
}

impl Decoder<'_> {
    /// The lossless part of the entropy decoder's `start_pass`: validate the scan
    /// header and derive the tables.
    pub(crate) fn start_lossless_pass(&mut self) -> R<()> {
        if !(1..=7).contains(&self.ss) || self.se != 0 || self.ah != 0 || self.al >= self.precision {
            return invalid(format!(
                "invalid lossless parameters Ss={} Se={} Ah={} Al={} (JERR_BAD_PROGRESSION, jdlossls.c:254-258)",
                self.ss, self.se, self.ah, self.al
            ));
        }
        if self.restart_interval != 0 && self.restart_interval as usize % self.mcus_per_row != 0 {
            return invalid(format!(
                "restart interval {} is not a multiple of the {} MCUs in a row (JERR_BAD_RESTART)",
                self.restart_interval, self.mcus_per_row
            ));
        }
        for ci in self.scan_comps.clone() {
            self.scanned[ci] = true;
            let dc = self.comps[ci].dc_tbl;
            self.derive_table(true, dc)?;
        }
        Ok(())
    }

    /// Decode one lossless scan into the sample planes.
    pub(crate) fn decode_scan_lossless(&mut self, planes: &mut [Vec<u8>]) -> R<()> {
        let n = self.scan_comps.len();
        let width_of = |s: &Self, slot: usize| -> usize {
            let c = &s.comps[s.scan_comps[slot]];
            if n == 1 {
                s.mcus_per_row
            } else {
                s.mcus_per_row * c.h
            }
        };
        let widths: Vec<usize> = (0..n).map(|k| width_of(self, k)).collect();
        // Sample rows of one iMCU row, per component slot: `v_samp_factor`, in an interleaved scan (one MCU row
        // holds them all) and in a non-interleaved one (it takes that many MCU rows of one sample row each).
        let rows_of: Vec<usize> = (0..n).map(|k| self.comps[self.scan_comps[k]].v).collect();
        let mcu_rows_per_imcu = if n == 1 { rows_of[0] } else { 1 };
        let mut diff: Vec<Vec<i32>> = (0..n).map(|k| vec![0; widths[k] * rows_of[k]]).collect();
        let mut prev: Vec<Vec<u16>> = widths.iter().map(|&w| vec![0; w]).collect();
        let mut cur: Vec<Vec<u16>> = widths.iter().map(|&w| vec![0; w]).collect();
        let mut first_row = vec![true; n];
        let rows_per_interval = if self.restart_interval == 0 {
            0
        } else {
            self.restart_interval as usize / self.mcus_per_row
        };
        let mut rows_to_go = rows_per_interval;
        let start = 1i32 << (self.precision - self.al - 1);
        let mut my = 0;
        while my < self.mcu_rows_in_scan {
            // `decompress_data` (jddiffct.c:175-204) decodes the iMCU row's MCU rows first and undifferences them
            // together afterwards, so a restart, or a row decoded from an exhausted segment, anywhere in the iMCU
            // row puts the whole iMCU row's first row back to the start-of-segment prediction (the entropy
            // decoder's `process_restart`, jdlhuff.c:166-188, and its out-of-data branch, jdlhuff.c:231-237, both
            // end in the undifferencer's `start_pass`, jdlossls.c:239-269).
            let mcu_rows = mcu_rows_per_imcu.min(self.mcu_rows_in_scan - my);
            let mut reset = false;
            for yoffset in 0..mcu_rows {
                if rows_per_interval != 0 && rows_to_go == 0 {
                    self.process_restart()?;
                    rows_to_go = rows_per_interval;
                    reset = true;
                }
                // A row that starts with the segment already out of data is all zero differences, predicted as
                // the first row of a segment: a flat 2^(P-1), not the row above.
                reset |= self.inp.insufficient;
                self.decode_difference_row(&mut diff, &widths, if n == 1 { yoffset } else { 0 })?;
                rows_to_go = rows_to_go.wrapping_sub(1);
            }
            if reset {
                first_row.iter_mut().for_each(|f| *f = true);
            }
            for slot in 0..n {
                let ci = self.scan_comps[slot];
                let stride = self.comps[ci].blocks_w;
                // Rows of this component present in the iMCU row: the decoded MCU rows of a non-interleaved scan, all
                // of the MCU's rows otherwise (those past the picture only land in the plane's padding).
                let present = if n == 1 { mcu_rows } else { rows_of[slot] };
                for dy in 0..present {
                    let w = widths[slot];
                    let d = &diff[slot][dy * w..(dy + 1) * w];
                    if first_row[slot] {
                        cur[slot][0] = ((d[0] + start) & 0xFFFF) as u16;
                        for x in 1..w {
                            cur[slot][x] = ((d[x] + i32::from(cur[slot][x - 1])) & 0xFFFF) as u16;
                        }
                        first_row[slot] = false;
                    } else {
                        undifference(self.ss, d, &prev[slot], &mut cur[slot]);
                    }
                    let y = if n == 1 { my + dy } else { my * rows_of[slot] + dy };
                    let row = &mut planes[ci][y * stride..y * stride + w];
                    for (o, &v) in row.iter_mut().zip(&cur[slot]) {
                        *o = (u32::from(v) << self.al) as u8;
                    }
                    std::mem::swap(&mut prev[slot], &mut cur[slot]);
                }
            }
            my += mcu_rows;
        }
        Ok(())
    }

    /// `decode_mcus` for one MCU row: the differences of every sample, in
    /// `diff[slot][row * width + x]`; `yoffset` is the MCU row's place in its iMCU row (a non-interleaved scan's
    /// sample row), 0 in an interleaved scan whose MCU carries all of its component's rows.
    fn decode_difference_row(&mut self, diff: &mut [Vec<i32>], widths: &[usize], yoffset: usize) -> R<()> {
        if self.inp.insufficient {
            // Out of data in this segment: the rest decodes as zero differences.
            if self.scan_comps.len() == 1 {
                let w = widths[0];
                diff[0][yoffset * w..(yoffset + 1) * w].fill(0);
            } else {
                diff.iter_mut().for_each(|d| d.fill(0));
            }
            return Ok(());
        }
        let nb = self.blocks_in_mcu;
        for mx in 0..self.mcus_per_row {
            for k in 0..nb {
                let (slot, dx, dy) = self.mcu_layout[k];
                let c = &self.comps[self.scan_comps[slot]];
                let Some(tbl) = self.dc_derived[c.dc_tbl].as_ref() else {
                    return invalid("internal: DC Huffman table not derived");
                };
                let inp = &mut self.inp;
                let s = inp.huff_decode(tbl)?;
                let value = if s == 0 {
                    0
                } else if s == 16 {
                    32768
                } else {
                    inp.check_bits(s)?;
                    let r = inp.get_bits(s);
                    huff_extend(r, s)
                };
                let x = if self.scan_comps.len() == 1 { mx } else { mx * c.h + dx };
                let row = if self.scan_comps.len() == 1 { yoffset } else { dy };
                diff[slot][row * widths[slot] + x] = value;
            }
        }
        Ok(())
    }
}

/// `jpeg_undifference2` to `7`, and `1`: reconstruct one sample row from its
/// differences, the previous reconstructed row and the predictor `psv`.
fn undifference(psv: usize, d: &[i32], prev: &[u16], cur: &mut [u16]) {
    cur[0] = ((d[0] + i32::from(prev[0])) & 0xFFFF) as u16;
    for x in 1..d.len() {
        let ra = i32::from(cur[x - 1]);
        let rb = i32::from(prev[x]);
        let rc = i32::from(prev[x - 1]);
        let p = match psv {
            1 => ra,
            2 => rb,
            3 => rc,
            4 => ra + rb - rc,
            5 => ra + ((rb - rc) >> 1),
            6 => rb + ((ra - rc) >> 1),
            _ => (ra + rb) >> 1,
        };
        cur[x] = ((d[x] + p) & 0xFFFF) as u16;
    }
}

/// Planes for a lossless frame: one byte per sample, the MCU grid wide.
pub(crate) fn lossless_store(d: &Decoder<'_>) -> Store {
    Store::Planes(d.comps.iter().map(|c| vec![0u8; c.blocks_w * c.blocks_h]).collect())
}

pub(crate) fn unsupported_precision(p: u32) -> JpegError {
    JpegError::Unsupported(format!(
        "lossless JPEG with {p}-bit samples: the reference's 8-bit decode path refuses it (JERR_BAD_PRECISION)"
    ))
}
