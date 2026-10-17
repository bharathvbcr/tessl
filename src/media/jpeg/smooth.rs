//! Block smoothing (`jdcoefct.c` `smoothing_ok`): libjpeg-turbo estimates the
//! first AC coefficients from neighbouring DC values when a progressive file
//! ends before they are fully known, and the reference does so by default.
//!
//! That estimator is not reproduced. A file that would trigger it is refused by
//! name rather than decoded to pixels the reference would not produce. Every
//! progressive file an encoder writes completes AC coefficients 1 to 5, so it
//! never reaches here; only a progressive stream cut short and closed with an
//! EOI (or one whose scans are damaged into that shape) does. A measurement of
//! how far a plain inverse transform is from the reference's smoothed output
//! on such files (max 3 to 73 per channel) is in `docs/embedgemma2.md`.

use super::Decoder;

impl Decoder<'_> {
    /// `smoothing_ok`: the reference smooths a progressive image when every
    /// component's quantization table is latched, DC and the first five AC
    /// quantizers are nonzero, every component's DC is at least partly known and
    /// at least one of AC coefficients 1 to 5 is not yet fully refined.
    pub(crate) fn smoothing_ok(&self) -> bool {
        if !self.progressive {
            return false;
        }
        let mut useful = false;
        for (ci, c) in self.comps.iter().enumerate() {
            let Some(q) = c.quant.as_ref() else {
                return false;
            };
            if [0usize, 1, 8, 16, 9, 2].iter().any(|&p| q[p] == 0) {
                return false;
            }
            let bits = &self.coef_bits[ci];
            if bits[0] < 0 {
                return false;
            }
            if (1..=5).any(|i| bits[i] != 0) {
                useful = true;
            }
        }
        useful
    }
}
