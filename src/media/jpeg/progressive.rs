//! Progressive Huffman decoding (`jdphuff.c`): the four scan kinds and the scan
//! parameter validation.

use super::huff::huff_extend;
use super::markers::NATURAL_ORDER;
use super::{invalid, Decoder, R};

impl Decoder<'_> {
    /// `start_pass_phuff_decoder`.
    pub(crate) fn start_progressive_pass(&mut self) -> R<()> {
        let is_dc_band = self.ss == 0;
        let mut bad = false;
        if is_dc_band {
            if self.se != 0 {
                bad = true;
            }
        } else {
            if self.ss > self.se || self.se >= 64 {
                bad = true;
            }
            if self.scan_comps.len() != 1 {
                bad = true;
            }
        }
        if self.ah != 0 && self.al != self.ah - 1 {
            bad = true;
        }
        if self.al > 13 {
            bad = true;
        }
        if bad {
            return invalid(format!(
                "bogus progression parameters Ss={} Se={} Ah={} Al={} (JERR_BAD_PROGRESSION)",
                self.ss, self.se, self.ah, self.al
            ));
        }
        self.record_progression();
        for k in 0..self.scan_comps.len() {
            let (dc, ac) = {
                let c = &self.comps[self.scan_comps[k]];
                (c.dc_tbl, c.ac_tbl)
            };
            if is_dc_band {
                if self.ah == 0 {
                    self.derive_table(true, dc)?;
                }
            } else {
                self.derive_table(false, ac)?;
            }
        }
        Ok(())
    }

    /// The progression bookkeeping of `start_pass` (Huffman and arithmetic alike).
    /// Inter-scan inconsistencies are warnings in libjpeg; the record is still
    /// updated, because block smoothing reads it. Before a scan overwrites
    /// `coef_bits`, the old values of zigzag positions `min(Ss, 1)..=max(Se, 9)`
    /// are kept (`0` for the first scan) as the record of the earlier state.
    pub(crate) fn record_progression(&mut self) {
        let lo = self.ss.min(1);
        // max(Se, 9) is at least 9, the last position kept.
        let hi = super::smooth::SAVED_COEFS - 1;
        let first_scan = self.input_scan_number <= 1;
        for &ci in &self.scan_comps {
            for k in lo..=hi {
                self.prev_bits[ci][k] = if first_scan { 0 } else { self.coef_bits[ci][k] };
            }
            for b in &mut self.coef_bits[ci][self.ss..=self.se] {
                *b = self.al as i32;
            }
        }
    }

    fn restart_if_due(&mut self) -> R<()> {
        if self.restart_interval != 0 && self.restarts_to_go == 0 {
            self.process_restart()?;
        }
        Ok(())
    }

    fn count_restart(&mut self) {
        if self.restart_interval != 0 {
            self.restarts_to_go = self.restarts_to_go.wrapping_sub(1);
        }
    }

    pub(crate) fn decode_mcu_dc_first(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        self.restart_if_due()?;
        if !self.inp.insufficient {
            for (block, &slot) in mcu.iter_mut().zip(&self.mcu_membership[..self.blocks_in_mcu]) {
                let c = &self.comps[self.scan_comps[slot]];
                let Some(tbl) = self.dc_derived[c.dc_tbl].as_ref() else {
                    return invalid("internal: DC Huffman table not derived");
                };
                let inp = &mut self.inp;
                let mut s = inp.huff_decode(tbl)?;
                if s != 0 {
                    inp.check_bits(s)?;
                    let r = inp.get_bits(s);
                    s = huff_extend(r, s);
                }
                let dc = s.wrapping_add(self.last_dc[slot]);
                self.last_dc[slot] = dc;
                block[0] = dc.wrapping_shl(self.al) as i16;
            }
        }
        self.count_restart();
        Ok(())
    }

    pub(crate) fn decode_mcu_ac_first(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        self.restart_if_due()?;
        if !self.inp.insufficient {
            let mut eobrun = self.eobrun;
            if eobrun > 0 {
                eobrun -= 1;
            } else {
                let c = &self.comps[self.scan_comps[0]];
                let Some(tbl) = self.ac_derived[c.ac_tbl].as_ref() else {
                    return invalid("internal: AC Huffman table not derived");
                };
                let inp = &mut self.inp;
                let block = &mut mcu[0];
                let mut k = self.ss;
                while k <= self.se {
                    let s = inp.huff_decode(tbl)?;
                    let mut r = s >> 4;
                    let s = s & 15;
                    if s != 0 {
                        k += r as usize;
                        inp.check_bits(s)?;
                        let bits = inp.get_bits(s);
                        let v = huff_extend(bits, s);
                        block[usize::from(NATURAL_ORDER[k])] = ((v as u32) << self.al) as i16;
                    } else if r == 15 {
                        k += 15;
                    } else {
                        eobrun = 1 << r;
                        if r != 0 {
                            inp.check_bits(r)?;
                            r = inp.get_bits(r);
                            eobrun += r as u32;
                        }
                        eobrun -= 1;
                        break;
                    }
                    k += 1;
                }
            }
            self.eobrun = eobrun;
        }
        self.count_restart();
        Ok(())
    }

    pub(crate) fn decode_mcu_dc_refine(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        self.restart_if_due()?;
        let p1 = 1i16 << self.al;
        for block in mcu.iter_mut().take(self.blocks_in_mcu) {
            self.inp.check_bits(1)?;
            if self.inp.get_bits(1) != 0 {
                block[0] |= p1;
            }
        }
        self.count_restart();
        Ok(())
    }

    pub(crate) fn decode_mcu_ac_refine(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        self.restart_if_due()?;
        if !self.inp.insufficient {
            let mut eobrun = self.eobrun;
            let c = &self.comps[self.scan_comps[0]];
            let Some(tbl) = self.ac_derived[c.ac_tbl].as_ref() else {
                return invalid("internal: AC Huffman table not derived");
            };
            let inp = &mut self.inp;
            let block = &mut mcu[0];
            let p1 = 1i16 << self.al;
            let m1 = (-1i16) << self.al;
            let mut k = self.ss;
            if eobrun == 0 {
                while k <= self.se {
                    let s = inp.huff_decode(tbl)?;
                    let mut r = s >> 4;
                    let mut s = s & 15;
                    if s != 0 {
                        // The size of a new coefficient should always be 1.
                        inp.check_bits(1)?;
                        s = if inp.get_bits(1) != 0 {
                            i32::from(p1)
                        } else {
                            i32::from(m1)
                        };
                    } else if r != 15 {
                        eobrun = 1 << r;
                        if r != 0 {
                            inp.check_bits(r)?;
                            r = inp.get_bits(r);
                            eobrun += r as u32;
                        }
                        break;
                    }
                    // Advance over already-nonzero coefficients and r still-zero
                    // ones, appending correction bits to the nonzeros.
                    loop {
                        let pos = usize::from(NATURAL_ORDER[k]);
                        let this = &mut block[pos];
                        if *this != 0 {
                            inp.check_bits(1)?;
                            if inp.get_bits(1) != 0 && (*this & p1) == 0 {
                                *this = this.wrapping_add(if *this >= 0 { p1 } else { m1 });
                            }
                        } else {
                            r -= 1;
                            if r < 0 {
                                break;
                            }
                        }
                        k += 1;
                        if k > self.se {
                            break;
                        }
                    }
                    if s != 0 {
                        block[usize::from(NATURAL_ORDER[k])] = s as i16;
                    }
                    k += 1;
                }
            }
            if eobrun > 0 {
                while k <= self.se {
                    let this = &mut block[usize::from(NATURAL_ORDER[k])];
                    if *this != 0 {
                        inp.check_bits(1)?;
                        if inp.get_bits(1) != 0 && (*this & p1) == 0 {
                            *this = this.wrapping_add(if *this >= 0 { p1 } else { m1 });
                        }
                    }
                    k += 1;
                }
                eobrun -= 1;
            }
            self.eobrun = eobrun;
        }
        self.count_restart();
        Ok(())
    }
}
