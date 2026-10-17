//! Arithmetic-coded JPEG (`jdarith.c`).

use super::{Decoder, JpegError, R};

#[derive(Default)]
pub(crate) struct State {}

impl Decoder<'_> {
    pub(crate) fn start_arith_pass(&mut self) -> R<()> {
        Err(JpegError::Unsupported("arithmetic-coded JPEG".into()))
    }

    pub(crate) fn decode_mcu_arith(&mut self, _mcu: &mut [[i16; 64]]) -> R<()> {
        Err(JpegError::Unsupported("arithmetic-coded JPEG".into()))
    }
}
