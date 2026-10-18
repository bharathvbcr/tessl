//! Block smoothing (`jdcoefct.c`: `smoothing_ok` and `decompress_smooth_data`).
//!
//! libjpeg-turbo estimates the low-frequency AC coefficients of a progressive
//! image from the DC values of the blocks around it when the scans it was given
//! do not carry them yet, and `do_block_smoothing` is on by default. The
//! reference (`torchvision.io.decode_image`) therefore smooths every progressive
//! file whose scans stop before AC coefficients 1 to 9 (zigzag order) are known
//! to their last bit: a stream cut short and closed with an EOI, or one that
//! only ever sent a DC scan. A complete progressive file never reaches the
//! estimator.
//!
//! # Provenance
//!
//! The method is ITU-T T.81 Annex K.8 (each estimated AC coefficient is
//! `Q00 * sum(w * DC) / (256 * Qxx)`, rounded half away from zero, over the DC
//! values of the neighbouring blocks, and limited to `2^Al - 1` in magnitude
//! while `Al > 0`). libjpeg-turbo 2.1 widened the neighbourhood from 3x3 to 5x5,
//! estimates nine AC coefficients instead of five, and, while no AC coefficient
//! has arrived at all, also replaces the DC by a smoothed value. No source for
//! those versions was available while this was written, so the weight tables
//! are *measured*, not recalled: each is an integer table read off the
//! reference by decoding progressive files that carry nothing but a DC scan
//! (and files whose AC scans are all-zero), one DC impulse at a time, with the
//! prediction of every block recovered exactly by matching the IDCT's pixels
//! (`scripts/gen_jpeg_fixtures.py` writes the same kind of files for the
//! committed fixtures). The tables, the loop structure (which neighbour rows are
//! replicated at the picture's edges, which `coef_bits` apply after a corrupt
//! scan) and every constant here are held by byte-equality with the reference
//! on the `smoothing_*` fixtures and on mutated files, not by a copy of the
//! library's code. The algorithm's description follows libjpeg's own: the
//! structure and the `smoothing_ok` conditions are what the library's
//! documentation and public behaviour specify; see `THIRD_PARTY_NOTICES.md`.

use super::idct;
use super::Decoder;

/// `SAVED_COEFS`: the zigzag positions 0 to 9 are the ones smoothing reads.
pub(crate) const SAVED_COEFS: usize = 10;

type Table = [[i32; 5]; 5];

/// Weights of the 25 DC values around a block (rows then columns, offset -2 to
/// +2) for the coefficient at natural position `0*8+1`, from the full (no AC
/// scan yet) estimate. Further tables are derived by transposition where the
/// coefficient is the mirror of another.
const FULL_01: Table = [
    [-1, -1, 0, 1, 1],
    [-3, 13, 0, -13, 3],
    [-3, 38, 0, -38, 3],
    [-3, 13, 0, -13, 3],
    [-1, -1, 0, 1, 1],
];
const FULL_20: Table = [
    [0, 0, 1, 0, 0],
    [0, 2, 7, 2, 0],
    [0, -5, -14, -5, 0],
    [0, 2, 7, 2, 0],
    [0, 0, 1, 0, 0],
];
const FULL_11: Table = [
    [-1, 0, 0, 0, 1],
    [0, 9, 0, -9, 0],
    [0, 0, 0, 0, 0],
    [0, -9, 0, 9, 0],
    [1, 0, 0, 0, -1],
];
const FULL_03: Table = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, -1, 0],
    [0, 2, 0, -2, 0],
    [0, 1, 0, -1, 0],
    [0, 0, 0, 0, 0],
];
const FULL_12: Table = [
    [0, 0, 0, 0, 0],
    [0, 1, -3, 1, 0],
    [0, 0, 0, 0, 0],
    [0, -1, 3, -1, 0],
    [0, 0, 0, 0, 0],
];
/// The estimate of the block's own DC when no AC scan has arrived.
const FULL_DC: Table = [
    [-2, -6, -8, -6, -2],
    [-6, 6, 42, 6, -6],
    [-8, 42, 152, 42, -8],
    [-6, 6, 42, 6, -6],
    [-2, -6, -8, -6, -2],
];

/// The estimates used once some AC data exists: only the neighbours in the
/// coefficient's own direction.
const PART_01: Table = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-7, 50, 0, -50, 7],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
];
const PART_20: Table = [
    [0, 0, -1, 0, 0],
    [0, 0, 13, 0, 0],
    [0, 0, -24, 0, 0],
    [0, 0, 13, 0, 0],
    [0, 0, -1, 0, 0],
];
const PART_11: Table = [
    [0, -1, 0, 1, 0],
    [-1, 10, 0, -10, 1],
    [0, 0, 0, 0, 0],
    [1, -10, 0, 10, -1],
    [0, 1, 0, -1, 0],
];

const fn transpose(t: &Table) -> Table {
    let mut o = [[0; 5]; 5];
    let mut i = 0;
    while i < 5 {
        let mut j = 0;
        while j < 5 {
            o[i][j] = t[j][i];
            j += 1;
        }
        i += 1;
    }
    o
}

const FULL_10: Table = transpose(&FULL_01);
const FULL_02: Table = transpose(&FULL_20);
const FULL_21: Table = transpose(&FULL_12);
const FULL_30: Table = transpose(&FULL_03);
const PART_10: Table = transpose(&PART_01);
const PART_02: Table = transpose(&PART_20);

/// One estimated coefficient: its natural position, its zigzag position (the
/// index into `coef_bits`) and its weights.
struct Est {
    nat: usize,
    zz: usize,
    full: &'static Table,
    part: Option<&'static Table>,
}

/// Evaluated in libjpeg's order: AC01, AC10, AC20, AC11, AC02, then (only while
/// no AC data exists) AC03, AC12, AC21, AC30.
static ESTIMATES: [Est; 9] = [
    Est {
        nat: 1,
        zz: 1,
        full: &FULL_01,
        part: Some(&PART_01),
    },
    Est {
        nat: 8,
        zz: 2,
        full: &FULL_10,
        part: Some(&PART_10),
    },
    Est {
        nat: 16,
        zz: 3,
        full: &FULL_20,
        part: Some(&PART_20),
    },
    Est {
        nat: 9,
        zz: 4,
        full: &FULL_11,
        part: Some(&PART_11),
    },
    Est {
        nat: 2,
        zz: 5,
        full: &FULL_02,
        part: Some(&PART_02),
    },
    Est {
        nat: 3,
        zz: 6,
        full: &FULL_03,
        part: None,
    },
    Est {
        nat: 10,
        zz: 7,
        full: &FULL_12,
        part: None,
    },
    Est {
        nat: 17,
        zz: 8,
        full: &FULL_21,
        part: None,
    },
    Est {
        nat: 24,
        zz: 9,
        full: &FULL_30,
        part: None,
    },
];

/// The quantizers `smoothing_ok` insists are nonzero (natural positions): DC,
/// AC01, AC10, AC20, AC11, AC02, AC03, AC12, AC21, AC30.
const GUARDED_QUANTIZERS: [usize; 10] = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24];

/// The progression record `smoothing_ok` latches: per component the current
/// `coef_bits[0..10]`, and the ones from before the last scan that touched the
/// component (used below the first corrupt row).
pub(crate) struct SmoothLatch {
    now: Vec<[i32; SAVED_COEFS]>,
    prev: Vec<[i32; SAVED_COEFS]>,
}

impl Decoder<'_> {
    /// `smoothing_ok`: the reference smooths a progressive image when every
    /// component's quantization table is latched with DC and AC01..AC30
    /// quantizers nonzero, every component's DC is at least partly known and at
    /// least one of zigzag coefficients 1 to 9 is not yet fully refined.
    pub(crate) fn smoothing_latch(&self) -> Option<SmoothLatch> {
        if !self.progressive {
            return None;
        }
        let mut now = Vec::with_capacity(self.comps.len());
        let mut prev = Vec::with_capacity(self.comps.len());
        let mut useful = false;
        for (ci, c) in self.comps.iter().enumerate() {
            let q = c.quant.as_ref()?;
            if GUARDED_QUANTIZERS.iter().any(|&p| q[p] == 0) {
                return None;
            }
            let bits = &self.coef_bits[ci];
            if bits[0] < 0 {
                return None;
            }
            let mut n = [0i32; SAVED_COEFS];
            let mut p = [-1i32; SAVED_COEFS];
            n[0] = bits[0];
            for k in 1..SAVED_COEFS {
                if self.input_scan_number > 1 {
                    p[k] = self.prev_bits[ci][k];
                }
                n[k] = bits[k];
                if bits[k] != 0 {
                    useful = true;
                }
            }
            now.push(n);
            prev.push(p);
        }
        useful.then_some(SmoothLatch { now, prev })
    }

    /// Inverse-transform component `ci` with block smoothing
    /// (`decompress_smooth_data`), `coefs` being its whole coefficient array.
    pub(crate) fn transform_smoothed(&self, ci: usize, coefs: &[i16], latch: &SmoothLatch) -> Vec<u8> {
        let c = &self.comps[ci];
        let stride = c.blocks_w * 8;
        let mut plane = vec![128u8; stride * c.blocks_h * 8];
        let Some(quant) = c.quant.as_ref() else {
            return plane;
        };
        let dc_at = |row: usize, col: usize| i32::from(coefs[(row * c.blocks_w + col) * 64]);
        let q00 = i64::from(quant[0]);
        let last_imcu = self.mcu_rows_full as i64 - 1;
        for by in 0..c.height_in_blocks {
            let rows = neighbour_rows(c.v, c.height_in_blocks, c.blocks_h, by, last_imcu);
            // Below the first iMCU row in which a scan ran out of data, the
            // progression record from before that scan applies.
            let imcu = (by / c.v) as i64;
            let bits = if imcu > self.last_good_imcu_row as i64 {
                &latch.prev[ci]
            } else {
                &latch.now[ci]
            };
            let change_dc = bits[1..SAVED_COEFS].iter().all(|&b| b == -1);
            for bx in 0..c.width_in_blocks {
                let at = (by * c.blocks_w + bx) * 64;
                let mut ws: [i16; 64] = coefs[at..at + 64].try_into().expect("64 coefficients");
                let mut win = [[0i32; 5]; 5];
                for (win_row, &row) in win.iter_mut().zip(&rows) {
                    for (k, dc) in win_row.iter_mut().enumerate() {
                        let col = (bx + k).saturating_sub(2).min(c.width_in_blocks - 1);
                        *dc = dc_at(row, col);
                    }
                }
                let sum = |t: &Table| -> i64 {
                    t.iter()
                        .zip(&win)
                        .flat_map(|(weights, dcs)| weights.iter().zip(dcs))
                        .map(|(&w, &dc)| i64::from(w) * i64::from(dc))
                        .sum()
                };
                let used = if change_dc { 9 } else { 5 };
                for e in &ESTIMATES[..used] {
                    let al = bits[e.zz];
                    if al != 0 && ws[e.nat] == 0 {
                        let table = if change_dc { e.full } else { e.part.unwrap_or(e.full) };
                        ws[e.nat] = predict(q00 * sum(table), i64::from(quant[e.nat]), al);
                    }
                }
                if change_dc {
                    ws[0] = predict(q00 * sum(&FULL_DC), q00, 0);
                }
                idct::islow(&ws, quant, &mut plane[by * 8 * stride + bx * 8..], stride);
            }
        }
        plane
    }
}

/// The five block rows `decompress_smooth_data` reads for block row `by` of a
/// component with `v` block rows per iMCU row, `height_in_blocks` real block
/// rows and `blocks_h` allocated ones (two above, this one, two below), with
/// libjpeg's handling of the top and bottom of the picture and of the iMCU-row
/// boundaries. `last_imcu` is the index of the picture's last iMCU row.
///
/// Provenance: the rule is measured, not copied. The oracle's output on
/// one-component pictures of every height from one block row to a few iMCU
/// rows past two, at 2, 3 and 4 block rows per iMCU row (`smoothing_vrows*`),
/// and on real 4:2:0, 4:4:0, 4:2:2 and taller-sampled pictures, fixes every
/// clause below; each is the simplest one all of those cases agree on.
fn neighbour_rows(v: usize, height_in_blocks: usize, blocks_h: usize, by: usize, last_imcu: i64) -> [usize; 5] {
    let imcu = (by / v) as i64;
    let brow = by % v;
    let block_rows = if imcu < last_imcu {
        v
    } else {
        match height_in_blocks % v {
            0 => v,
            m => m,
        }
    };
    let prev = if brow > 0 || imcu > 0 { by - 1 } else { by };
    // Two rows up: inside the iMCU row, or from the second iMCU row on. In the
    // second iMCU row the row two up lies in the first, unless that row is the
    // picture's last one and holds a single block row (then it is the row one
    // up, as in the first iMCU row).
    let lone_last_row = imcu == last_imcu && block_rows == 1;
    let prev_prev = if brow > 1 || imcu > 1 || (imcu == 1 && by > 1 && !lone_last_row) {
        by - 2
    } else {
        prev
    };
    let next = if brow + 1 < block_rows || imcu < last_imcu {
        by + 1
    } else {
        by
    };
    // Two rows down: inside the iMCU row, or in any iMCU row but the last, where
    // the row below may be one of the padding rows that complete the last iMCU
    // row (they hold the blocks an interleaved scan decoded there, or zeros) and
    // exist only for components with more than one block row per iMCU row.
    let next_next = if brow + 2 < block_rows || (imcu < last_imcu && by + 2 < blocks_h) {
        by + 2
    } else {
        next
    };
    [prev_prev, prev, by, next, next_next]
}

/// One estimate in `K.8`'s form: `(Q << 7) + num) / (Q << 8)` on the magnitude,
/// limited to `2^Al - 1` while `Al > 0`, then given the sign of `num`, and stored
/// in the 16-bit coefficient.
fn predict(num: i64, q: i64, al: i32) -> i16 {
    let (mag, negative) = if num >= 0 {
        (((q << 7) + num) / (q << 8), false)
    } else {
        (((q << 7) - num) / (q << 8), true)
    };
    let mut pred = mag as i32;
    if al > 0 && pred >= (1 << al) {
        pred = (1 << al) - 1;
    }
    if negative {
        pred = pred.wrapping_neg();
    }
    pred as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(t: &Table) -> i32 {
        t.iter().flatten().sum()
    }

    #[test]
    fn the_weights_of_a_derivative_estimate_sum_to_zero_and_the_dc_estimate_to_one() {
        // A flat neighbourhood has no gradient or curvature and keeps its DC.
        for e in &ESTIMATES {
            assert_eq!(sum(e.full), 0, "full table of natural position {}", e.nat);
            if let Some(p) = e.part {
                assert_eq!(sum(p), 0, "partial table of natural position {}", e.nat);
            }
        }
        assert_eq!(sum(&FULL_DC), 256);
    }

    /// Every row `neighbour_rows` names lies in the allocated block rows (the
    /// reader indexes the coefficient array with it), whatever the sampling
    /// factor and height.
    #[test]
    fn the_neighbour_rows_stay_inside_the_allocated_rows() {
        for v in 1..=4usize {
            for height in 1..=4 * v + 3 {
                let imcus = height.div_ceil(v);
                let last = imcus as i64 - 1;
                for by in 0..height {
                    for row in neighbour_rows(v, height, imcus * v, by, last) {
                        assert!(row < imcus * v, "v {v}, height {height}, row {by}: {row}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_neighbour_rows_follow_the_measured_rule() {
        // Two block rows per iMCU row, five block rows (imcus of 2, 2, 1): the last iMCU row
        // holds one real row, so row 4 sees its own row where it has no rows below.
        assert_eq!(neighbour_rows(2, 5, 6, 0, 2), [0, 0, 0, 1, 2]);
        assert_eq!(neighbour_rows(2, 5, 6, 1, 2), [0, 0, 1, 2, 3]);
        assert_eq!(neighbour_rows(2, 5, 6, 2, 2), [0, 1, 2, 3, 4]);
        assert_eq!(neighbour_rows(2, 5, 6, 3, 2), [1, 2, 3, 4, 5]);
        assert_eq!(neighbour_rows(2, 5, 6, 4, 2), [2, 3, 4, 4, 4]);
        // Four rows in two iMCU rows: the second is the last and holds two real rows.
        assert_eq!(neighbour_rows(2, 4, 4, 2, 1), [0, 1, 2, 3, 3]);
        // Three rows in two iMCU rows: the last holds one, whose row two up is the row one up.
        assert_eq!(neighbour_rows(2, 3, 4, 2, 1), [1, 1, 2, 2, 2]);
        // One block row per iMCU row: the top row repeats itself above, and two rows down are
        // reachable while two iMCU rows remain.
        assert_eq!(neighbour_rows(1, 3, 3, 0, 2), [0, 0, 0, 1, 2]);
        assert_eq!(neighbour_rows(1, 3, 3, 1, 2), [0, 0, 1, 2, 2]);
    }

    #[test]
    fn an_estimate_rounds_half_away_from_zero_and_is_limited_by_the_point_transform() {
        // (Q << 7) + num) / (Q << 8) on the magnitude: half a quantizer step rounds up.
        assert_eq!(predict(0, 6, -1), 0);
        assert_eq!(predict(6 * 128, 6, -1), 1);
        assert_eq!(predict(6 * 128 - 1, 6, -1), 0);
        assert_eq!(predict(-6 * 128, 6, -1), -1);
        assert_eq!(predict(-(6 * 128 - 1), 6, -1), 0);
        // Al > 0 caps the magnitude at 2^Al - 1; Al of 0 or -1 does not.
        assert_eq!(predict(6 * 256 * 100, 6, 3), 7);
        assert_eq!(predict(-6 * 256 * 100, 6, 3), -7);
        assert_eq!(predict(6 * 256 * 100, 6, 0), 100);
        assert_eq!(predict(6 * 256 * 100, 6, -1), 100);
        // The coefficient is a 16-bit value: an enormous estimate wraps as the C cast does.
        assert_eq!(predict(65535 * 256 * 70000, 65535, -1), 70000i32 as i16);
    }
}
