//! Lossless JPEG (SOF3, ITU-T T.81 Annex H) as libjpeg-turbo 3.x decodes it
//! (`jdlossls.c`, `jdlhuff.c`, `jddiffct.c`, `jdpred.c`, `jdscale.c`).
//!
//! A lossless scan codes, for every sample, the difference to a prediction made
//! from the samples to its left (`Ra`), above (`Rb`) and above-left (`Rc`):
//! the Huffman code names the difference's category (0 to 16) and the extra bits
//! give its value, exactly as a DC difference does in the DCT modes. What
//! follows is the reference's behaviour, established with the oracle on the
//! fixtures `lossless_*` and not by a copy of the library:
//!
//! * the unit of coding is one sample, so a "block" is a sample, an MCU holds
//!   `h * v` samples of each component, and the picture is `ceil(w / h_max)` MCUs
//!   wide;
//! * the first sample of a scan, and of every restart interval, is predicted
//!   from `2^(P - Pt - 1)` and the rest of its row from the left; the first sample
//!   of a later row is predicted from above; everything else by the scan's
//!   predictor selection value (`Ss`, 1 to 7). Restart intervals are whole MCU
//!   rows;
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
                "invalid lossless parameters Ss={} Se={} Ah={} Al={} (JERR_BAD_LOSSLESS)",
                self.ss, self.se, self.ah, self.al
            ));
        }
        if self.restart_interval != 0 && self.restart_interval as usize % self.mcus_per_row != 0 {
            return invalid(format!(
                "restart interval {} is not a multiple of the {} MCUs in a row (JERR_BAD_RESTART)",
                self.restart_interval, self.mcus_per_row
            ));
        }
        // In several scans the reference undifferences the whole frame after the last scan, counting restart
        // rows in that scan's MCU rows; with components of unequal size that disagrees with the scan's own
        // count, and the combination is not reproduced (it is refused, not guessed).
        if self.restart_interval != 0 && self.scan_comps.len() < self.comps.len() {
            let first = &self.comps[0];
            if self
                .comps
                .iter()
                .any(|c| (c.width_in_blocks, c.height_in_blocks) != (first.width_in_blocks, first.height_in_blocks))
            {
                return Err(JpegError::Unsupported(
                    "lossless JPEG in several scans with a restart interval and components of different sizes: \
                     the reference counts the restart rows over the last scan's MCU rows, which this decoder does \
                     not reproduce"
                        .into(),
                ));
            }
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
        // Sample rows of one MCU row, per component slot.
        let rows_of: Vec<usize> = (0..n)
            .map(|k| if n == 1 { 1 } else { self.comps[self.scan_comps[k]].v })
            .collect();
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
        for my in 0..self.mcu_rows_in_scan {
            if rows_per_interval != 0 && rows_to_go == 0 {
                self.process_restart()?;
                rows_to_go = rows_per_interval;
                first_row.iter_mut().for_each(|f| *f = true);
            }
            // A row that starts with the segment already out of data is all zero differences, and the
            // reference then predicts it as the first row of a segment: a flat 2^(P-1), not the row above.
            if self.inp.insufficient {
                first_row.iter_mut().for_each(|f| *f = true);
            }
            self.decode_difference_row(&mut diff, &widths)?;
            for slot in 0..n {
                let ci = self.scan_comps[slot];
                let stride = self.comps[ci].blocks_w;
                for dy in 0..rows_of[slot] {
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
                    let y = my * rows_of[slot] + dy;
                    let row = &mut planes[ci][y * stride..y * stride + w];
                    for (o, &v) in row.iter_mut().zip(&cur[slot]) {
                        *o = (u32::from(v) << self.al) as u8;
                    }
                    std::mem::swap(&mut prev[slot], &mut cur[slot]);
                }
            }
            rows_to_go = rows_to_go.wrapping_sub(1);
        }
        Ok(())
    }

    /// `decode_mcus` for one MCU row: the differences of every sample, in
    /// `diff[slot][row * width + x]`.
    fn decode_difference_row(&mut self, diff: &mut [Vec<i32>], widths: &[usize]) -> R<()> {
        if self.inp.insufficient {
            // Out of data in this segment: the rest decodes as zero differences.
            diff.iter_mut().for_each(|d| d.fill(0));
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
                diff[slot][dy * widths[slot] + x] = value;
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
