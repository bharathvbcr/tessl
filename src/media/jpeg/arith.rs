//! Arithmetic-coded JPEG (`jdarith.c`): the QM-style decoder of T.81 Annex D
//! and the four scan kinds built on it. The reference decodes these files, and a
//! decoder that refused them would regress against it.
//!
//! Unlike the Huffman decoder, hitting a marker inside the data is legal here:
//! zeros are supplied until decoding ends. A magnitude or spectral overflow puts
//! the decoder in an error state (`ct == -1`) in which the remaining MCUs of the
//! segment are left untouched, until a restart clears it.

use super::markers::NATURAL_ORDER;
use super::{invalid, Decoder, Input, R};

/// `jpeg_aritab`: Table D.3 packed as `(Qe << 16) | (Next_Index_MPS << 8) |
/// (Switch_MPS << 7) | Next_Index_LPS`, 113 states and the fixed-probability
/// state 113. Read from the reference's own library, compared with T.81.
const ARITAB: [u32; 114] = [
    0x5a1d0181, 0x2586020e, 0x11140310, 0x080b0412, 0x03d80514, 0x01da0617, 0x00e50719, 0x006f081c, 0x0036091e,
    0x001a0a21, 0x000d0b23, 0x00060c09, 0x00030d0a, 0x00010d0c, 0x5a7f0f8f, 0x3f251024, 0x2cf21126, 0x207c1227,
    0x17b91328, 0x1182142a, 0x0cef152b, 0x09a1162d, 0x072f172e, 0x055c1830, 0x04061931, 0x03031a33, 0x02401b34,
    0x01b11c36, 0x01441d38, 0x00f51e39, 0x00b71f3b, 0x008a203c, 0x0068213e, 0x004e223f, 0x003b2320, 0x002c0921,
    0x5ae125a5, 0x484c2640, 0x3a0d2741, 0x2ef12843, 0x261f2944, 0x1f332a45, 0x19a82b46, 0x15182c48, 0x11772d49,
    0x0e742e4a, 0x0bfb2f4b, 0x09f8304d, 0x0861314e, 0x0706324f, 0x05cd3330, 0x04de3432, 0x040f3532, 0x03633633,
    0x02d43734, 0x025c3835, 0x01f83936, 0x01a43a37, 0x01603b38, 0x01253c39, 0x00f63d3a, 0x00cb3e3b, 0x00ab3f3d,
    0x008f203d, 0x5b1241c1, 0x4d044250, 0x412c4351, 0x37d84452, 0x2fe84553, 0x293c4654, 0x23794756, 0x1edf4857,
    0x1aa94957, 0x174e4a48, 0x14244b48, 0x119c4c4a, 0x0f6b4d4a, 0x0d514e4b, 0x0bb64f4d, 0x0a40304d, 0x583251d0,
    0x4d1c5258, 0x438e5359, 0x3bdd545a, 0x34ee555b, 0x2eae565c, 0x299a575d, 0x25164756, 0x557059d8, 0x4ca95a5f,
    0x44d95b60, 0x3e225c61, 0x38245d63, 0x32b45e63, 0x2e17565d, 0x56a860df, 0x4f466165, 0x47e56266, 0x41cf6367,
    0x3c3d6468, 0x375e5d63, 0x52316669, 0x4c0f676a, 0x4639686b, 0x415e6367, 0x56276ae9, 0x50e76b6c, 0x4b85676d,
    0x55976d6e, 0x504f6b6f, 0x5a106fee, 0x55226d70, 0x59eb6ff0, 0x5a1d7171,
];

const DC_STAT_BINS: usize = 64;
const AC_STAT_BINS: usize = 256;
const NUM_ARITH_TBLS: usize = 16;

/// The register state of the decoder (`c`, `a`, `ct` of `jdarith.c`).
pub(crate) struct Coder {
    c: i64,
    a: i64,
    /// Bits left in the bit part of `c`: -16 at start, 0..=7 while running, -1
    /// after an overflow.
    ct: i32,
}

impl Coder {
    fn reset(&mut self) {
        self.c = 0;
        self.a = 0;
        self.ct = -16;
    }

    /// `get_byte`: the next input byte, or the data end the reference refuses.
    fn next_data(inp: &mut Input) -> R<u8> {
        inp.src.byte()
    }

    /// `arith_decode`: one binary decision with the adaptive state `st`.
    fn decode(&mut self, inp: &mut Input, st: &mut u8) -> R<u8> {
        while self.a < 0x8000 {
            self.ct -= 1;
            if self.ct < 0 {
                let data: u8 = if inp.unread_marker != 0 {
                    0
                } else {
                    let mut d = Self::next_data(inp)?;
                    if d == 0xFF {
                        loop {
                            d = Self::next_data(inp)?;
                            if d != 0xFF {
                                break;
                            }
                        }
                        if d == 0 {
                            0xFF
                        } else {
                            inp.unread_marker = d;
                            0
                        }
                    } else {
                        d
                    }
                };
                self.c = (self.c << 8) | i64::from(data);
                self.ct += 8;
                if self.ct < 0 {
                    self.ct += 1;
                    if self.ct == 0 {
                        self.a = 0x8000;
                    }
                }
            }
            self.a <<= 1;
        }
        let mut sv = i32::from(*st);
        let mut qe = i64::from(ARITAB[(sv & 0x7F) as usize]);
        let nl = (qe & 0xFF) as u8;
        qe >>= 8;
        let nm = (qe & 0xFF) as u8;
        qe >>= 8;
        let mut temp = self.a - qe;
        self.a = temp;
        temp <<= self.ct;
        if self.c >= temp {
            self.c -= temp;
            // Conditional LPS exchange: when the MPS interval is the smaller one
            // the upper region is the MPS.
            if self.a < qe {
                self.a = qe;
                *st = (sv & 0x80) as u8 ^ nm;
            } else {
                self.a = qe;
                *st = (sv & 0x80) as u8 ^ nl;
                sv ^= 0x80;
            }
        } else if self.a < 0x8000 {
            if self.a < qe {
                *st = (sv & 0x80) as u8 ^ nl;
                sv ^= 0x80;
            } else {
                *st = (sv & 0x80) as u8 ^ nm;
            }
        }
        Ok((sv >> 7) as u8)
    }
}

pub(crate) struct State {
    coder: Coder,
    dc_stats: Vec<[u8; DC_STAT_BINS]>,
    ac_stats: Vec<[u8; AC_STAT_BINS]>,
    last_dc_val: [i32; 4],
    dc_context: [usize; 4],
}

impl Default for State {
    fn default() -> Self {
        State {
            coder: Coder { c: 0, a: 0, ct: -16 },
            dc_stats: vec![[0; DC_STAT_BINS]; NUM_ARITH_TBLS],
            ac_stats: vec![[0; AC_STAT_BINS]; NUM_ARITH_TBLS],
            last_dc_val: [0; 4],
            dc_context: [0; 4],
        }
    }
}

impl Decoder<'_> {
    /// Zero the statistics areas and DC predictions the scan uses.
    fn arith_reset_stats(&mut self) {
        let progressive = self.progressive;
        for k in 0..self.scan_comps.len() {
            let c = &self.comps[self.scan_comps[k]];
            let (dc, ac) = (c.dc_tbl, c.ac_tbl);
            if !progressive || (self.ss == 0 && self.ah == 0) {
                self.arith.dc_stats[dc] = [0; DC_STAT_BINS];
                self.arith.last_dc_val[k] = 0;
                self.arith.dc_context[k] = 0;
            }
            if !progressive || self.ss != 0 {
                self.arith.ac_stats[ac] = [0; AC_STAT_BINS];
            }
        }
    }

    /// `start_pass` of the arithmetic decoder.
    pub(crate) fn start_arith_pass(&mut self) -> R<()> {
        if self.progressive {
            let mut bad = false;
            if self.ss == 0 {
                bad |= self.se != 0;
            } else {
                bad |= self.se < self.ss || self.se > 63 || self.scan_comps.len() != 1;
            }
            if self.ah != 0 && self.ah - 1 != self.al {
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
            for &ci in &self.scan_comps {
                for b in &mut self.coef_bits[ci][self.ss..=self.se] {
                    *b = self.al as i32;
                }
            }
        }
        self.arith_reset_stats();
        self.arith.coder.reset();
        Ok(())
    }

    /// `process_restart` of the arithmetic decoder: take the marker, reset the
    /// statistics and registers (which also clears an error state).
    fn arith_process_restart(&mut self) -> R<()> {
        self.read_restart_marker()?;
        self.arith_reset_stats();
        self.arith.coder.reset();
        self.restarts_to_go = self.restart_interval;
        Ok(())
    }

    pub(crate) fn decode_mcu_arith(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        if self.restart_interval != 0 {
            if self.restarts_to_go == 0 {
                self.arith_process_restart()?;
            }
            self.restarts_to_go = self.restarts_to_go.wrapping_sub(1);
        }
        if !self.progressive {
            return self.arith_sequential(mcu);
        }
        match (self.ss == 0, self.ah == 0) {
            (true, true) => self.arith_dc_first(mcu),
            (true, false) => self.arith_dc_refine(mcu),
            (false, true) => self.arith_ac_first(mcu),
            (false, false) => self.arith_ac_refine(mcu),
        }
    }

    /// Decode one DC difference with the context of component slot `k`, table
    /// `tbl` (Figures F.19 to F.24). `Ok(None)` is the overflow error state.
    fn arith_dc_diff(&mut self, k: usize, tbl: usize) -> R<Option<i32>> {
        let a = &mut self.arith;
        let inp = &mut self.inp;
        let ctx = a.dc_context[k];
        if a.coder.decode(inp, &mut a.dc_stats[tbl][ctx])? == 0 {
            a.dc_context[k] = 0;
            return Ok(Some(0));
        }
        let sign = a.coder.decode(inp, &mut a.dc_stats[tbl][ctx + 1])? as usize;
        let mut st = ctx + 2 + sign;
        let mut m: i32 = i32::from(a.coder.decode(inp, &mut a.dc_stats[tbl][st])?);
        if m != 0 {
            st = 20;
            while a.coder.decode(inp, &mut a.dc_stats[tbl][st])? != 0 {
                m <<= 1;
                if m == 0x8000 {
                    a.coder.ct = -1;
                    return Ok(None);
                }
                st += 1;
            }
        }
        let l = u32::from(self.arith_dc_l[tbl]);
        let u = u32::from(self.arith_dc_u[tbl]);
        a.dc_context[k] = if m < ((1i64 << l) >> 1) as i32 {
            0
        } else if m > ((1i64 << u) >> 1) as i32 {
            12 + sign * 4
        } else {
            4 + sign * 4
        };
        let mut v = m;
        st += 14;
        loop {
            m >>= 1;
            if m == 0 {
                break;
            }
            if a.coder.decode(inp, &mut a.dc_stats[tbl][st])? != 0 {
                v |= m;
            }
        }
        v += 1;
        if sign != 0 {
            v = -v;
        }
        Ok(Some(v))
    }

    fn arith_dc(&mut self, k: usize, tbl: usize) -> R<Option<i32>> {
        match self.arith_dc_diff(k, tbl)? {
            None => Ok(None),
            Some(v) => {
                let a = &mut self.arith;
                a.last_dc_val[k] = (a.last_dc_val[k] + v) & 0xffff;
                Ok(Some(a.last_dc_val[k]))
            }
        }
    }

    /// The AC run/size decode shared by the sequential and progressive-first
    /// scans for coefficient index `k` (Figures F.20 to F.24). Returns the value
    /// and the new `k`, or `None` for the error state, or `Some((None, k))` at
    /// the end of block.
    fn arith_ac_coef(&mut self, tbl: usize, k: usize, se: usize) -> R<AcStep> {
        let a = &mut self.arith;
        let inp = &mut self.inp;
        let mut k = k;
        let mut st = 3 * (k - 1);
        if a.coder.decode(inp, &mut a.ac_stats[tbl][st])? != 0 {
            return Ok(AcStep::Eob);
        }
        while a.coder.decode(inp, &mut a.ac_stats[tbl][st + 1])? == 0 {
            st += 3;
            k += 1;
            if k > se {
                a.coder.ct = -1;
                return Ok(AcStep::Overflow);
            }
        }
        let sign = a.coder.decode(inp, &mut 113u8)?;
        st += 2;
        let mut m: i32 = i32::from(a.coder.decode(inp, &mut a.ac_stats[tbl][st])?);
        if m != 0 && a.coder.decode(inp, &mut a.ac_stats[tbl][st])? != 0 {
            m <<= 1;
            st = if k <= usize::from(self.arith_ac_k[tbl]) {
                189
            } else {
                217
            };
            while a.coder.decode(inp, &mut a.ac_stats[tbl][st])? != 0 {
                m <<= 1;
                if m == 0x8000 {
                    a.coder.ct = -1;
                    return Ok(AcStep::Overflow);
                }
                st += 1;
            }
        }
        let mut v = m;
        st += 14;
        loop {
            m >>= 1;
            if m == 0 {
                break;
            }
            if a.coder.decode(inp, &mut a.ac_stats[tbl][st])? != 0 {
                v |= m;
            }
        }
        v += 1;
        if sign != 0 {
            v = -v;
        }
        Ok(AcStep::Coef(v, k))
    }

    fn arith_sequential(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        if self.arith.coder.ct == -1 {
            return Ok(());
        }
        for blkn in 0..self.blocks_in_mcu {
            let k = self.mcu_membership[blkn];
            let c = &self.comps[self.scan_comps[k]];
            let (dct, act) = (c.dc_tbl, c.ac_tbl);
            let Some(dc) = self.arith_dc(k, dct)? else {
                return Ok(());
            };
            mcu[blkn][0] = dc as i16;
            let mut kk = 1usize;
            while kk <= 63 {
                match self.arith_ac_coef(act, kk, 63)? {
                    AcStep::Eob => break,
                    AcStep::Overflow => return Ok(()),
                    AcStep::Coef(v, at) => {
                        mcu[blkn][usize::from(NATURAL_ORDER[at])] = v as i16;
                        kk = at + 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn arith_dc_first(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        if self.arith.coder.ct == -1 {
            return Ok(());
        }
        for blkn in 0..self.blocks_in_mcu {
            let k = self.mcu_membership[blkn];
            let dct = self.comps[self.scan_comps[k]].dc_tbl;
            let Some(dc) = self.arith_dc(k, dct)? else {
                return Ok(());
            };
            mcu[blkn][0] = dc.wrapping_shl(self.al) as i16;
        }
        Ok(())
    }

    fn arith_dc_refine(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        let p1 = 1i16 << self.al;
        for blk in mcu.iter_mut().take(self.blocks_in_mcu) {
            let a = &mut self.arith;
            if a.coder.decode(&mut self.inp, &mut 113u8)? != 0 {
                blk[0] |= p1;
            }
        }
        Ok(())
    }

    fn arith_ac_first(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        if self.arith.coder.ct == -1 {
            return Ok(());
        }
        let tbl = self.comps[self.scan_comps[0]].ac_tbl;
        let mut k = self.ss;
        while k <= self.se {
            match self.arith_ac_coef(tbl, k, self.se)? {
                AcStep::Eob => break,
                AcStep::Overflow => return Ok(()),
                AcStep::Coef(v, at) => {
                    mcu[0][usize::from(NATURAL_ORDER[at])] = ((v as u32) << self.al) as i16;
                    k = at + 1;
                }
            }
        }
        Ok(())
    }

    fn arith_ac_refine(&mut self, mcu: &mut [[i16; 64]]) -> R<()> {
        if self.arith.coder.ct == -1 {
            return Ok(());
        }
        let tbl = self.comps[self.scan_comps[0]].ac_tbl;
        let block = &mut mcu[0];
        let p1 = 1i16 << self.al;
        let m1 = (-1i16) << self.al;
        let mut kex = self.se;
        while kex > 0 {
            if block[usize::from(NATURAL_ORDER[kex])] != 0 {
                break;
            }
            kex -= 1;
        }
        let a = &mut self.arith;
        let inp = &mut self.inp;
        let mut k = self.ss;
        while k <= self.se {
            let mut st = 3 * (k - 1);
            if k > kex && a.coder.decode(inp, &mut a.ac_stats[tbl][st])? != 0 {
                break;
            }
            loop {
                let this = &mut block[usize::from(NATURAL_ORDER[k])];
                if *this != 0 {
                    if a.coder.decode(inp, &mut a.ac_stats[tbl][st + 2])? != 0 {
                        *this = this.wrapping_add(if *this < 0 { m1 } else { p1 });
                    }
                    break;
                }
                if a.coder.decode(inp, &mut a.ac_stats[tbl][st + 1])? != 0 {
                    *this = if a.coder.decode(inp, &mut 113u8)? != 0 { m1 } else { p1 };
                    break;
                }
                st += 3;
                k += 1;
                if k > self.se {
                    a.coder.ct = -1;
                    return Ok(());
                }
            }
            k += 1;
        }
        Ok(())
    }
}

enum AcStep {
    Eob,
    Overflow,
    Coef(i32, usize),
}
