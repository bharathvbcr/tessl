//! Block smoothing of progressive files whose low-frequency coefficients are
//! not all known (`jdcoefct.c` `smoothing_ok` and `decompress_smooth_data`).

use super::{Decoder, JpegError, R};

impl Decoder<'_> {
    /// `smoothing_ok`: libjpeg smooths a progressive image when every
    /// component's DC is known, the first quantizers are nonzero and at least
    /// one of AC coefficients 1 to 5 is not yet fully refined.
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

    pub(crate) fn smooth_component(&self, _ci: usize, _coefs: &[i16], _plane: &mut [u8]) -> R<()> {
        Err(JpegError::Unsupported(
            "progressive JPEG that ends before its low-frequency coefficients are complete (block smoothing)".into(),
        ))
    }
}
