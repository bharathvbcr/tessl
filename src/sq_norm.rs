//! The sum of squares of a set of f32 or bf16 windows: the square of the
//! global L2 norm `torch.nn.utils.clip_grad_norm_` takes over their values.
//!
//! Two dispatches for any number of windows (`kernels/qwen35_adamw.metal`):
//! `qwen35_sq_sum_pieces` sums the squares of every window in pieces of at
//! most [`SQ_PIECE`] elements through a table of GPU addresses, and
//! `qwen35_sum_exact_f32` adds the pieces exactly in integer arithmetic,
//! which [`decode_sq_norm`] rounds to f64 once. Each piece is scaled by a
//! power of two set by its own largest element before it is squared, so a
//! finite input never overflows or underflows a square: the result is finite
//! for any finite values, within `2^-18` relative of the exact sum, and
//! deterministic (the same bits whatever the windows' order or grouping). A
//! NaN gives NaN and an infinity (with no NaN) gives +inf.
//!
//! [`SqNormPlan`] validates the windows and builds the table on the host;
//! [`SqNormPlan::encode`] records both dispatches into caller-owned buffers
//! without waiting, so a caller with its own command stream and scratch pool
//! (`ojas-metal`'s `clip_grad_norm`) waits where it already waits.
//! [`sq_norm`] is the allocate, encode, wait and decode in one call.
//! [`crate::qwen35_model::Qwen35Model::grad_sq_norm`] is that over a model's
//! gradients.

use crate::dispatch::{set_gpu_buf_offset, set_u32};
use crate::nn::require_runtime;
use crate::qwen35_params::Window;
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::{DType, GpuBuffer};

use objc2_metal::MTLBuffer;

/// Bytes of one entry of `qwen35_sq_sum_pieces`'s table (`Qwen35SqSumSlot`
/// in `kernels/qwen35_adamw.metal`, which asserts the same size): the
/// window's GPU address, then eight `u32`s.
const SQ_SLOT_BYTES: usize = 40;

/// Elements of one `qwen35_sq_sum_pieces` piece (`QWEN35_SQ_PIECE`).
pub const SQ_PIECE: usize = 4096;

/// Most pieces `qwen35_sum_exact_f32` adds: below 2^27, its 128-bit sum of
/// values below 2^100 units keeps its top bit clear. About 5.5e11 elements.
pub const SQ_MAX_PIECES: u64 = (1 << 27) - 1;

/// Words `qwen35_sum_exact_f32` writes (see [`decode_sq_norm`]).
pub const SQ_OUT_WORDS: usize = 6;

/// One window of values: `rows` x `width` at leading dimension `ld` from
/// element `off`, of `buf` starting `byte_off` bytes in, f32 or bf16.
#[derive(Clone, Copy)]
pub struct SqWindow<'a> {
    pub buf: &'a GpuBuffer,
    pub byte_off: usize,
    pub dtype: DType,
    pub rows: usize,
    pub width: usize,
    pub ld: usize,
    pub off: usize,
}

impl<'a> SqWindow<'a> {
    /// `n` contiguous elements of `buf` from `byte_off`.
    pub fn dense(buf: &'a GpuBuffer, byte_off: usize, dtype: DType, n: usize) -> Self {
        Window::dense(buf, byte_off, dtype, n).into()
    }

    fn window(&self) -> Window<'a> {
        Window {
            buf: self.buf,
            byte_off: self.byte_off,
            dtype: self.dtype,
            rows: self.rows,
            width: self.width,
            ld: self.ld,
            off: self.off,
        }
    }
}

impl<'a> From<Window<'a>> for SqWindow<'a> {
    fn from(w: Window<'a>) -> Self {
        SqWindow {
            buf: w.buf,
            byte_off: w.byte_off,
            dtype: w.dtype,
            rows: w.rows,
            width: w.width,
            ld: w.ld,
            off: w.off,
        }
    }
}

/// A checked set of windows and its slot table.
pub struct SqNormPlan {
    words: Vec<u32>,
    slots: u32,
    pieces: u32,
}

impl SqNormPlan {
    /// Every window lies inside its buffer, is f32 or bf16, is not empty,
    /// fits the kernels' `u32`s and belongs to `rt`; together they hold at
    /// most [`SQ_MAX_PIECES`] pieces. `what` labels errors. An empty set is
    /// a plan with no pieces, whose sum is 0 and which encodes nothing.
    pub fn new(rt: &GpuRuntime, windows: &[SqWindow<'_>], what: &str) -> Result<Self, String> {
        let mut words = Vec::with_capacity(windows.len() * SQ_SLOT_BYTES / 4);
        let mut pieces = 0u64;
        for (i, sw) in windows.iter().enumerate() {
            let w = sw.window();
            w.check(format_args!("{what}: window {i}"))?;
            // The kernel reads it by address alone, so nothing binds it: only
            // this runtime's residency set makes it visible to the GPU.
            require_runtime(rt, w.buf, format_args!("{what}: window {i}"))?;
            let per_row = w.width.div_ceil(SQ_PIECE);
            let first = pieces;
            pieces += (w.rows * per_row) as u64;
            if pieces > SQ_MAX_PIECES {
                return Err(format!(
                    "{what}: the windows exceed {SQ_MAX_PIECES} pieces of {SQ_PIECE}"
                ));
            }
            let addr = w.buf.metal().gpuAddress().wrapping_add(w.byte_off as u64);
            // Every extent fits u32 (`check`), and `first` < SQ_MAX_PIECES.
            words.extend([
                addr as u32,
                (addr >> 32) as u32,
                w.rows as u32,
                w.width as u32,
                w.ld as u32,
                w.off as u32,
                per_row as u32,
                first as u32,
                u32::from(w.dtype == DType::BF16),
                0,
            ]);
        }
        debug_assert_eq!(words.len() * 4, windows.len() * SQ_SLOT_BYTES);
        let slots = u32::try_from(windows.len()).map_err(|_| format!("{what}: {} windows", windows.len()))?;
        Ok(SqNormPlan {
            words,
            slots,
            pieces: pieces as u32,
        })
    }

    /// The slot table, to be written to a buffer for [`Self::encode`].
    pub fn table_words(&self) -> &[u32] {
        &self.words
    }

    /// Pieces of [`SQ_PIECE`] the windows cut into: `parts` holds two words
    /// per piece.
    pub fn pieces(&self) -> usize {
        self.pieces as usize
    }

    /// Record both dispatches in one binder scope (one commit, so one wait
    /// whatever the encode mode), without waiting: `table` holds
    /// [`Self::table_words`]; `parts` has room for `2 * pieces()` words
    /// (each piece's sum, then its shift), every one written here; `out`
    /// for [`SQ_OUT_WORDS`], read by [`decode_sq_norm`] once the work is
    /// done. The windows' buffers must stay alive until then: the table
    /// names them by address, so nothing here retains them. A plan with no
    /// pieces records nothing and leaves `out` alone.
    pub fn encode(
        &self,
        rt: &GpuRuntime,
        table: (&GpuBuffer, usize),
        parts: (&GpuBuffer, usize),
        out: (&GpuBuffer, usize),
        what: &str,
    ) -> Result<(), String> {
        if self.pieces == 0 {
            return Ok(());
        }
        let n = self.pieces as usize;
        let ranges = [
            ("table", table, self.words.len() * 4),
            ("parts", parts, 8 * n),
            ("out", out, 4 * SQ_OUT_WORDS),
        ];
        for (name, (buf, off), bytes) in ranges {
            require_runtime(rt, buf, format_args!("{what}: {name}"))?;
            if off % 4 != 0 || off.checked_add(bytes).is_none_or(|end| end > buf.nbytes()) {
                return Err(format!(
                    "{what}: {name} needs {bytes} bytes at offset {off} of a {}-byte buffer",
                    buf.nbytes()
                ));
            }
        }
        // The finish reads `parts` while it writes `out`, and the pieces read
        // `table` while they write `parts`.
        for (i, &(a, (ab, ao), an)) in ranges.iter().enumerate() {
            for &(b, (bb, bo), bn) in &ranges[i + 1..] {
                if ab.aliases(bb) && ao < bo + bn && bo < ao + an {
                    return Err(format!("{what}: {a} and {b} overlap"));
                }
            }
        }
        let shifts = parts.1 + 4 * n;
        let (p_pieces, p_sum) = (
            rt.pipeline("qwen35_sq_sum_pieces")?,
            rt.pipeline("qwen35_sum_exact_f32")?,
        );
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&p_pieces);
            set_gpu_buf_offset(bnd, table.0, table.1, 0);
            set_u32(bnd, self.slots, 1);
            set_gpu_buf_offset(bnd, parts.0, parts.1, 2);
            set_gpu_buf_offset(bnd, parts.0, shifts, 3);
            bnd.dispatch(mtl_size(n, 1, 1), mtl_size(256, 1, 1));
            // The binder's barrier after the first orders the second.
            bnd.set_pipeline(&p_sum);
            set_gpu_buf_offset(bnd, parts.0, parts.1, 0);
            set_gpu_buf_offset(bnd, out.0, out.1, 1);
            set_u32(bnd, self.pieces, 2);
            set_gpu_buf_offset(bnd, parts.0, shifts, 3);
            bnd.dispatch(mtl_size(1, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })
    }
}

/// The sum of squares over `windows` ([`SqNormPlan`]): allocates, encodes,
/// waits for the runtime, and decodes. 0 for no windows, with no dispatch.
pub fn sq_norm(rt: &GpuRuntime, windows: &[SqWindow<'_>], what: &str) -> Result<f64, String> {
    let plan = SqNormPlan::new(rt, windows, what)?;
    if plan.pieces() == 0 {
        return Ok(0.0);
    }
    // Fresh, so written without waiting for the GPU; every word of `parts`
    // and `out` is written by the kernels.
    let table = rt.alloc_buffer_from_u32(plan.table_words())?;
    let parts = rt.alloc_buffer(8 * plan.pieces())?;
    let out = rt.alloc_buffer(4 * SQ_OUT_WORDS)?;
    plan.encode(rt, (&table, 0), (&parts, 0), (&out, 0), what)?;
    rt.synchronize()?;
    let r = out.try_contents_u32()?;
    decode_sq_norm(&r[..SQ_OUT_WORDS])
}

/// What `qwen35_sum_exact_f32` wrote, as f64: words 0..4 (least significant
/// first) are the exact sum in units of `2^(etop - 226)`, word 4 the largest
/// piece's bits, word 5 `etop` (an `i32`). The integer is rounded to f64
/// once; the scaling is by a power of two, so exact. A word 4 that is not
/// finite is the result: NaN for any NaN, else +inf.
pub fn decode_sq_norm(out: &[u32]) -> Result<f64, String> {
    let w: &[u32; SQ_OUT_WORDS] = out
        .try_into()
        .map_err(|_| format!("decode_sq_norm: {} words, not {SQ_OUT_WORDS}", out.len()))?;
    Ok(exact_sum_f64([w[0], w[1], w[2], w[3]], w[4], w[5] as i32))
}

fn exact_sum_f64(words: [u32; 4], top: u32, etop: i32) -> f64 {
    if top > 0x7f80_0000 {
        return f64::NAN;
    }
    if top == 0x7f80_0000 {
        return f64::INFINITY;
    }
    let sum = words.iter().rev().fold(0u128, |acc, &w| (acc << 32) | u128::from(w));
    sum as f64 * 2f64.powi(etop - 226)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `qwen35_sum_exact_f32`'s words read back as the 128-bit sum in units
    /// of `2^(etop - 226)`, rounded to f64 once (to nearest, ties to even);
    /// a largest value that is not finite is the result.
    #[test]
    fn exact_sum_words_round_once() {
        let one = 1.0f32.to_bits();
        // 1.0 at its own exponent (127) is 2^99 units: bit 3 of the top word.
        let at = |bit: u32| {
            let mut w = [0u32; 4];
            w[(bit / 32) as usize] |= 1 << (bit % 32);
            w
        };
        let or = |a: [u32; 4], b: [u32; 4]| [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]];
        assert_eq!(exact_sum_f64(at(99), one, 127), 1.0);
        assert_eq!(exact_sum_f64(or(at(99), at(47)), one, 127), 1.0 + 2f64.powi(-52));
        // Below f64's resolution at 1: to nearest, a tie to even.
        assert_eq!(exact_sum_f64(or(at(99), at(39)), one, 127), 1.0);
        assert_eq!(exact_sum_f64(or(at(99), at(46)), one, 127), 1.0);
        assert_eq!(
            exact_sum_f64(or(or(at(99), at(47)), at(46)), one, 127),
            1.0 + 2f64.powi(-51)
        );
        assert_eq!(exact_sum_f64([u32::MAX; 4], one, 127), 2f64.powi(29));
        // A subnormal largest value counts as exponent 1.
        assert_eq!(exact_sum_f64(at(76), 1, 1), 2f64.powi(-149));
        assert_eq!(exact_sum_f64([0; 4], 0, 1), 0.0);
        // A piece shifted past f32's range: 1.0 * 2^252 and 1.0 * 2^-252.
        assert_eq!(exact_sum_f64(at(99), one, 127 + 252), 2f64.powi(252));
        assert_eq!(exact_sum_f64(at(99), one, 127 - 252), 2f64.powi(-252));
        assert_eq!(exact_sum_f64([0; 4], 0x7f80_0000, 0), f64::INFINITY);
        assert!(exact_sum_f64([0; 4], 0x7fc0_0000, 0).is_nan());
        assert!(exact_sum_f64([0; 4], 0xffc0_0000, 0).is_nan());
    }

    #[test]
    fn decode_refuses_a_short_or_long_read() {
        assert!(decode_sq_norm(&[0; 5]).unwrap_err().contains("5 words"));
        assert!(decode_sq_norm(&[0; 7]).unwrap_err().contains("7 words"));
        assert_eq!(decode_sq_norm(&[0, 0, 0, 0, 0, 1]).unwrap(), 0.0);
    }
}
