//! Huffman tables and the bit reader (`jdhuff.c`): `jpeg_make_d_derived_tbl`,
//! `jpeg_fill_bit_buffer` and `jpeg_huff_decode`, including what they do when
//! the data is damaged.

use super::{invalid, Input, R};

/// A table as the DHT marker stores it: counts per code length 1..=16 and the
/// symbols in code order.
#[derive(Clone)]
pub(crate) struct HuffSpec {
    pub bits: [u8; 17],
    pub vals: [u8; 256],
}

/// Decoding tables derived from a [`HuffSpec`].
#[derive(Clone)]
pub(crate) struct Derived {
    pub maxcode: [i32; 18],
    pub valoffset: [i32; 18],
    pub vals: [u8; 256],
    /// `(code length << 8) | symbol` for every 8-bit prefix; length 9 marks a
    /// code longer than the lookahead.
    pub lookup: [u16; 256],
}

pub(crate) const HUFF_LOOKAHEAD: i32 = 8;
/// `BIT_BUF_SIZE - 7` with a 64-bit bit buffer.
pub(crate) const MIN_GET_BITS: i32 = 57;

impl Derived {
    /// `jpeg_make_d_derived_tbl`. `is_dc` tables may only name symbols 0..=15.
    pub fn build(spec: &HuffSpec, is_dc: bool, tblno: usize) -> R<Derived> {
        let mut huffsize = [0u8; 257];
        let mut p = 0usize;
        for l in 1..=16usize {
            let i = usize::from(spec.bits[l]);
            if p + i > 256 {
                return invalid(format!(
                    "Huffman table {tblno} has more than 256 codes (JERR_BAD_HUFF_TABLE)"
                ));
            }
            for _ in 0..i {
                huffsize[p] = l as u8;
                p += 1;
            }
        }
        huffsize[p] = 0;
        let numsymbols = p;
        let mut huffcode = [0u32; 257];
        let mut code = 0u32;
        let mut si = i32::from(huffsize[0]);
        let mut p = 0usize;
        while huffsize[p] != 0 {
            while i32::from(huffsize[p]) == si {
                huffcode[p] = code;
                p += 1;
                code += 1;
            }
            if i64::from(code) >= (1i64 << si) {
                return invalid(format!(
                    "Huffman table {tblno} is not a legal code tree (JERR_BAD_HUFF_TABLE)"
                ));
            }
            code <<= 1;
            si += 1;
        }
        let mut maxcode = [0i32; 18];
        let mut valoffset = [0i32; 18];
        let mut p = 0usize;
        for l in 1..=16usize {
            if spec.bits[l] != 0 {
                valoffset[l] = p as i32 - huffcode[p] as i32;
                p += usize::from(spec.bits[l]);
                maxcode[l] = huffcode[p - 1] as i32;
            } else {
                maxcode[l] = -1;
            }
        }
        valoffset[17] = 0;
        maxcode[17] = 0xF_FFFF;
        let mut lookup = [((HUFF_LOOKAHEAD + 1) as u16) << 8; 256];
        let mut p = 0usize;
        for l in 1..=HUFF_LOOKAHEAD as usize {
            for _ in 0..spec.bits[l] {
                let lookbits = (huffcode[p] as usize) << (HUFF_LOOKAHEAD as usize - l);
                for ctr in 0..(1usize << (HUFF_LOOKAHEAD as usize - l)) {
                    lookup[lookbits + ctr] = ((l as u16) << 8) | u16::from(spec.vals[p]);
                }
                p += 1;
            }
        }
        if is_dc {
            for &sym in &spec.vals[..numsymbols] {
                if sym > 15 {
                    return invalid(format!(
                        "DC Huffman table {tblno} names symbol {sym} (JERR_BAD_HUFF_TABLE)"
                    ));
                }
            }
        }
        Ok(Derived {
            maxcode,
            valoffset,
            vals: spec.vals,
            lookup,
        })
    }
}

const fn spec(counts: [u8; 16], vals: &[u8]) -> HuffSpec {
    let mut bits = [0u8; 17];
    let mut i = 0;
    while i < 16 {
        bits[i + 1] = counts[i];
        i += 1;
    }
    let mut v = [0u8; 256];
    let mut j = 0;
    while j < vals.len() {
        v[j] = vals[j];
        j += 1;
    }
    HuffSpec { bits, vals: v }
}

const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// The tables of T.81 Annex K.3 that libjpeg-turbo installs for any of the four
/// tables 0 and 1 a sequential Huffman stream does not define itself (Motion
/// JPEG frames omit them): DC luminance, DC chrominance, AC luminance, AC
/// chrominance.
pub(crate) fn standard_tables() -> [HuffSpec; 4] {
    [
        spec([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], &DC_VALS),
        spec([0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], &DC_VALS),
        spec(
            [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
            &[
                0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22,
                0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33,
                0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34,
                0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55,
                0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76,
                0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
                0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5,
                0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
                0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1,
                0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
            ],
        ),
        spec(
            [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
            &[
                0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13,
                0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62,
                0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29,
                0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54,
                0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
                0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94,
                0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3,
                0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
                0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
                0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
            ],
        ),
    ]
}

/// `HUFF_EXTEND`.
#[inline]
pub(crate) fn huff_extend(x: i32, s: i32) -> i32 {
    if x < (1 << (s - 1)) {
        x + ((-1i32 << s) + 1)
    } else {
        x
    }
}

impl Input<'_> {
    /// `jpeg_fill_bit_buffer`: read bytes until `MIN_GET_BITS` bits are held or a
    /// marker ends the entropy-coded data. If the marker arrives with fewer than
    /// `nbits` bits held, the missing bits are zeros and the segment is marked
    /// out of data.
    pub fn fill_bit_buffer(&mut self, nbits: i32) -> R<()> {
        if self.unread_marker == 0 {
            while self.bits_left < MIN_GET_BITS {
                let mut c = self.src.byte()?;
                if c == 0xFF {
                    loop {
                        c = self.src.byte()?;
                        if c != 0xFF {
                            break;
                        }
                    }
                    if c == 0 {
                        c = 0xFF;
                    } else {
                        self.unread_marker = c;
                        return self.no_more_bytes(nbits);
                    }
                }
                self.acc = (self.acc << 8) | u64::from(c);
                self.bits_left += 8;
            }
            Ok(())
        } else {
            self.no_more_bytes(nbits)
        }
    }

    fn no_more_bytes(&mut self, nbits: i32) -> R<()> {
        if nbits > self.bits_left {
            self.insufficient = true;
            self.acc <<= MIN_GET_BITS - self.bits_left;
            self.bits_left = MIN_GET_BITS;
        }
        Ok(())
    }

    /// `CHECK_BIT_BUFFER`.
    #[inline]
    pub fn check_bits(&mut self, nbits: i32) -> R<()> {
        if self.bits_left < nbits {
            self.fill_bit_buffer(nbits)?;
        }
        Ok(())
    }

    /// `GET_BITS` (the bits are known to be present).
    #[inline]
    pub fn get_bits(&mut self, nbits: i32) -> i32 {
        self.bits_left -= nbits;
        ((self.acc >> self.bits_left) & ((1u64 << nbits) - 1)) as i32
    }

    /// `HUFF_DECODE`.
    #[inline]
    pub fn huff_decode(&mut self, t: &Derived) -> R<i32> {
        let mut nb = 1;
        let mut slow = false;
        if self.bits_left < HUFF_LOOKAHEAD {
            self.fill_bit_buffer(0)?;
            if self.bits_left < HUFF_LOOKAHEAD {
                slow = true;
            }
        }
        if !slow {
            let look = ((self.acc >> (self.bits_left - HUFF_LOOKAHEAD)) & 0xFF) as usize;
            let e = t.lookup[look];
            nb = i32::from(e >> 8);
            if nb <= HUFF_LOOKAHEAD {
                self.bits_left -= nb;
                return Ok(i32::from(e & 0xFF));
            }
        }
        self.huff_decode_slow(t, nb)
    }

    /// `jpeg_huff_decode`: the rest of a code one bit at a time. A code that
    /// matches nothing (length 17) decodes as symbol 0.
    fn huff_decode_slow(&mut self, t: &Derived, min_bits: i32) -> R<i32> {
        let mut l = min_bits;
        self.check_bits(l)?;
        let mut code = self.get_bits(l);
        while code > t.maxcode[l as usize] {
            code <<= 1;
            self.check_bits(1)?;
            code |= self.get_bits(1);
            l += 1;
        }
        if l > 16 {
            return Ok(0);
        }
        Ok(i32::from(t.vals[((code + t.valoffset[l as usize]) & 0xFF) as usize]))
    }
}
