//! The scan loop shared by every coding, and the sequential Huffman MCU decoder
//! (`jdhuff.c` `decode_mcu`).

use super::huff::{huff_extend, Derived};
use super::idct;
use super::markers::NATURAL_ORDER;
use super::{invalid, Decoder, JpegError, Store, D_MAX_BLOCKS_IN_MCU, MAX_COMPS_IN_SCAN, NUM_TBLS, R};

impl Decoder<'_> {
    /// `jpeg_make_d_derived_tbl` for one table.
    pub(crate) fn derive_table(&mut self, is_dc: bool, tblno: usize) -> R<()> {
        let spec = if tblno < NUM_TBLS {
            if is_dc {
                self.dc_spec[tblno].as_ref()
            } else {
                self.ac_spec[tblno].as_ref()
            }
        } else {
            None
        };
        let Some(spec) = spec else {
            return invalid(format!("Huffman table {tblno} is not defined (JERR_NO_HUFF_TABLE)"));
        };
        let d = Derived::build(spec, is_dc, tblno)?;
        if is_dc {
            self.dc_derived[tblno] = Some(d);
        } else {
            self.ac_derived[tblno] = Some(d);
        }
        Ok(())
    }

    /// The entropy decoder's `start_pass`: derive the tables and reset the bit
    /// reader, the DC predictions and the restart counter.
    pub(crate) fn start_entropy_pass(&mut self) -> R<()> {
        self.mcu_layout.clear();
        let n = self.scan_comps.len();
        if n == 1 {
            self.mcu_layout.push((0, 0, 0));
        } else {
            for k in 0..n {
                let c = &self.comps[self.scan_comps[k]];
                for y in 0..c.v {
                    for x in 0..c.h {
                        self.mcu_layout.push((k, x, y));
                    }
                }
            }
        }
        self.inp.acc = 0;
        self.inp.bits_left = 0;
        self.inp.insufficient = false;
        self.last_dc = [0; MAX_COMPS_IN_SCAN];
        self.eobrun = 0;
        self.restarts_to_go = self.restart_interval;
        if self.arithmetic {
            return self.start_arith_pass();
        }
        if self.progressive {
            return self.start_progressive_pass();
        }
        for k in 0..n {
            let (dc, ac) = {
                let c = &self.comps[self.scan_comps[k]];
                (c.dc_tbl, c.ac_tbl)
            };
            self.derive_table(true, dc)?;
            self.derive_table(false, ac)?;
        }
        Ok(())
    }

    /// Decode the scan whose header was just read, storing its blocks.
    pub(crate) fn decode_scan(&mut self) -> R<()> {
        let mut store = self
            .store
            .take()
            .ok_or_else(|| JpegError::Invalid("internal: no block store".into()))?;
        let r = self.scan_loop(&mut store);
        self.store = Some(store);
        r
    }

    fn scan_loop(&mut self, store: &mut Store) -> R<()> {
        let nb = self.blocks_in_mcu;
        let mut mcu = [[0i16; 64]; D_MAX_BLOCKS_IN_MCU];
        for my in 0..self.mcu_rows_in_scan {
            for mx in 0..self.mcus_per_row {
                match store {
                    Store::Planes(_) => {
                        for b in &mut mcu[..nb] {
                            *b = [0; 64];
                        }
                    }
                    Store::Coefs(coefs) => {
                        for k in 0..nb {
                            let (ci, bx, by) = self.block_pos(k, mx, my);
                            let at = (by * self.comps[ci].blocks_w + bx) * 64;
                            mcu[k].copy_from_slice(&coefs[ci][at..at + 64]);
                        }
                    }
                }
                self.decode_mcu(&mut mcu[..nb])?;
                match store {
                    Store::Planes(planes) => {
                        for k in 0..nb {
                            let (ci, bx, by) = self.block_pos(k, mx, my);
                            let c = &self.comps[ci];
                            if bx < c.width_in_blocks && by < c.height_in_blocks {
                                let quant = c.quant.as_ref().ok_or_else(|| {
                                    JpegError::Invalid("internal: component without a quantization table".into())
                                })?;
                                let stride = c.blocks_w * 8;
                                idct::islow(&mcu[k], quant, &mut planes[ci][by * 8 * stride + bx * 8..], stride);
                            }
                        }
                    }
                    Store::Coefs(coefs) => {
                        for k in 0..nb {
                            let (ci, bx, by) = self.block_pos(k, mx, my);
                            let at = (by * self.comps[ci].blocks_w + bx) * 64;
                            coefs[ci][at..at + 64].copy_from_slice(&mcu[k]);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Component index and block column and row of block `k` of the MCU at
    /// `(mx, my)`.
    #[inline]
    fn block_pos(&self, k: usize, mx: usize, my: usize) -> (usize, usize, usize) {
        let (slot, dx, dy) = self.mcu_layout[k];
        let ci = self.scan_comps[slot];
        let c = &self.comps[ci];
        (ci, mx * c.mcu_width + dx, my * c.mcu_height + dy)
    }

    fn decode_mcu(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        if self.arithmetic {
            return self.decode_mcu_arith(mcu);
        }
        if !self.progressive {
            return self.decode_mcu_sequential(mcu);
        }
        match (self.ss == 0, self.ah == 0) {
            (true, true) => self.decode_mcu_dc_first(mcu),
            (true, false) => self.decode_mcu_dc_refine(mcu),
            (false, true) => self.decode_mcu_ac_first(mcu),
            (false, false) => self.decode_mcu_ac_refine(mcu),
        }
    }

    /// `process_restart`: discard the bit buffer, take the restart marker (or
    /// resynchronize), reset the predictions and the run, and clear the
    /// out-of-data flag unless a marker is still pending.
    pub(crate) fn process_restart(&mut self) -> R<()> {
        self.inp.bits_left = 0;
        self.read_restart_marker()?;
        self.last_dc = [0; MAX_COMPS_IN_SCAN];
        self.eobrun = 0;
        self.restarts_to_go = self.restart_interval;
        if self.inp.unread_marker == 0 {
            self.inp.insufficient = false;
        }
        Ok(())
    }

    fn decode_mcu_sequential(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        if self.restart_interval != 0 && self.restarts_to_go == 0 {
            self.process_restart()?;
        }
        if !self.inp.insufficient {
            for blkn in 0..self.blocks_in_mcu {
                let slot = self.mcu_membership[blkn];
                let c = &self.comps[self.scan_comps[slot]];
                let (Some(dt), Some(at)) = (self.dc_derived[c.dc_tbl].as_ref(), self.ac_derived[c.ac_tbl].as_ref())
                else {
                    return invalid("internal: Huffman table not derived");
                };
                let block = &mut mcu[blkn];
                let inp = &mut self.inp;
                let mut s = inp.huff_decode(dt)?;
                if s != 0 {
                    inp.check_bits(s)?;
                    let r = inp.get_bits(s);
                    s = huff_extend(r, s);
                }
                let dc = s.wrapping_add(self.last_dc[slot]);
                self.last_dc[slot] = dc;
                block[0] = dc as i16;
                let mut k = 1usize;
                while k < 64 {
                    let s = inp.huff_decode(at)?;
                    let r = (s >> 4) as usize;
                    let s = s & 15;
                    if s != 0 {
                        k += r;
                        inp.check_bits(s)?;
                        let bits = inp.get_bits(s);
                        block[usize::from(NATURAL_ORDER[k])] = huff_extend(bits, s) as i16;
                    } else {
                        if r != 15 {
                            break;
                        }
                        k += 15;
                    }
                    k += 1;
                }
            }
        }
        if self.restart_interval != 0 {
            self.restarts_to_go = self.restarts_to_go.wrapping_sub(1);
        }
        Ok(())
    }
}
