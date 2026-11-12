//! MLX-affine dequantisation to bf16: the elementwise kernel under a
//! "dequantise to a reused scratch, then the existing GEMM" path.
//!
//! `kernels/dequant_affine.metal` expands a quantised matrix in MLX's
//! row-major layout into a bf16 matrix:
//!
//! ```text
//! out[r, c] = bf16_rne( f32(scale[r, c / G]) * f32(q[r, c]) + f32(bias[r, c / G]) )
//! ```
//!
//! with one rounding, which is what `mx.dequantize` does on the GPU stream
//! (and the form a host oracle must use: `bf16(bf16(s * q) + b)` is wrong).
//!
//! # Layout
//!
//! For a [`QuantShape`] `{ rows, cols, group_size }` and `bits` in {4, 8}:
//!
//! - `packed`: `u32` `[rows, cols * bits / 32]`. A q8 word holds 4 codes and a
//!   q4 word 8; code `i` of a word sits at bits `[bits * i, bits * (i + 1))`.
//! - `scales`, `biases`: bf16 `[rows, cols / group_size]`, separate buffers
//!   (MLX's, unlike [`crate::nn::Q4MlxBank`]'s interleaved pairs).
//! - `out`: bf16 `[rows, cols]`, caller-owned. It may be larger than the matrix
//!   (a scratch reused across layers of different shapes); only the first
//!   `rows * cols` elements are written and the rest are left alone.
//!
//! `group_size` is 32, 64 or 128 (MLX's domain below 256; the kernel takes the
//! group as a shift). Anything else is refused.
//!
//! # Non-finite results: the fault word
//!
//! The kernel stores what the arithmetic gives: a value beyond the bf16 range
//! is `+-inf` (as in [`crate::gemm::cast_f32_to_bf16`]), and a non-finite scale
//! or bias propagates by IEEE rules (a NaN is the GPU's canonical `0x7fc0`).
//! It also reports: every thread whose output holds an infinity or a NaN stores
//! `1` to the caller's `fault` word (one `u32` shared by every thread; unlike
//! `i8_quantize_rows`, whose `bad` flag is per row, it is sticky across dispatches).
//! The kernel never clears it, and it stores rather than ORs: a faulting
//! thread overwrites the word with `1` (a caller's own non-zero code does not
//! survive it), and a clean dequant leaves the word as it found it. So give
//! the dequants of one run a word of their own. A caller zeroes it once, lets
//! any number of dequants store into it, and reads it once after a
//! [`GpuRuntime::synchronize`]: non-zero means some output matrix holds a
//! non-finite value and must not be used. The flag is sticky by design, so a
//! clean dequant cannot hide an earlier bad one. The dequant itself never
//! reads the word back (that is a host wait); [`check_dequant_affine_fault`]
//! is the checked read for the caller's own sync point.
//!
//! For finite inputs, a sufficient condition for a finite output is
//! `|scale| * qmax + |bias| <= bf16::MAX` (qmax 255 or 15); a caller that
//! refuses non-finite weights at load can check the static scales and biases
//! against it instead.
//!
//! # Subnormals
//!
//! The device runs with fast math, which flushes f32 subnormals: a subnormal
//! scale, bias or result behaves as a zero of the same sign. This differs from
//! the IEEE host formula, and matches `mx.dequantize(stream=mx.gpu)`
//! (`tests/dequant_affine.rs` pins both facts). The multiply-add is fused: the
//! exact product is added before the one rounding to f32, which matters only
//! where `scale * q` alone would overflow f32.

use std::sync::Arc;

use crate::dispatch::{dispatch_1d, set_gpu_buf, set_u32};
use crate::nn::{require, require_disjoint_writes, QuantShape};
use crate::runtime::GpuRuntime;
use crate::tensor::GpuBuffer;

/// Group sizes the kernel accepts. Each is a power of two (the kernel indexes a
/// group by shift) and a multiple of the 8 codes of a q4 word, so one word
/// never straddles two groups.
pub const DEQUANT_AFFINE_GROUP_SIZES: [u32; 3] = [32, 64, 128];

// `plan` relies on this in place of a whole-word check on `cols * bits`: `cols %
// group_size == 0` and `8 | group_size` give `8 | cols`, so `cols * 4` and
// `cols * 8` are whole numbers of 32-bit words. All the proof needs is `8 | g`;
// the assertion is stricter on purpose: a power of two (the kernel indexes a
// group by shift) of at least 32, the smallest size the kernel is documented and
// tested for, so a smaller size such as 8 or 16 cannot enter the list without
// someone re-reading this reasoning and the kernel's.
const _: () = {
    let mut i = 0;
    while i < DEQUANT_AFFINE_GROUP_SIZES.len() {
        let g = DEQUANT_AFFINE_GROUP_SIZES[i];
        assert!(g.is_power_of_two() && g >= 32 && g.is_multiple_of(8));
        i += 1;
    }
};

/// Code width of a packed matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineBits {
    /// 4-bit codes, 8 per `u32`.
    Q4,
    /// 8-bit codes, 4 per `u32`.
    Q8,
}

impl AffineBits {
    /// Bits per code.
    pub fn bits(self) -> u32 {
        match self {
            AffineBits::Q4 => 4,
            AffineBits::Q8 => 8,
        }
    }

    /// The kernel entry point.
    pub fn entry(self) -> &'static str {
        match self {
            AffineBits::Q4 => "dequant_affine_q4_bf16",
            AffineBits::Q8 => "dequant_affine_q8_bf16",
        }
    }
}

/// A quantised matrix in MLX's row-major affine layout (see the module docs).
#[derive(Clone, Copy)]
pub struct DequantAffineBank<'a> {
    /// Packed codes, `u32` `[rows, cols * bits / 32]`.
    pub packed: &'a GpuBuffer,
    /// bf16 `[rows, cols / group_size]`.
    pub scales: &'a GpuBuffer,
    /// bf16 `[rows, cols / group_size]`.
    pub biases: &'a GpuBuffer,
}

/// What a validated call dispatches.
struct Plan {
    /// `u32` words of packed codes, i.e. threads.
    words: usize,
    /// `log2(group_size)`.
    group_shift: u32,
}

fn product(dims: &[usize], what: &str) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |p, &d| p.checked_mul(d))
        .ok_or_else(|| format!("{what}: dimension product overflows usize"))
}

fn plan(
    rt: &GpuRuntime,
    bits: AffineBits,
    bank: &DequantAffineBank<'_>,
    out: &GpuBuffer,
    fault: &GpuBuffer,
    shape: &QuantShape,
) -> Result<Plan, String> {
    let entry = bits.entry();
    if !DEQUANT_AFFINE_GROUP_SIZES.contains(&shape.group_size) {
        return Err(format!(
            "{entry}: group_size {} is outside the supported set {DEQUANT_AFFINE_GROUP_SIZES:?}",
            shape.group_size
        ));
    }
    if !shape.cols.is_multiple_of(shape.group_size) {
        return Err(format!(
            "{entry}: cols {} is not a multiple of group_size {}; the ragged tail group would be dropped",
            shape.cols, shape.group_size
        ));
    }
    let (rows, cols) = (shape.rows as usize, shape.cols as usize);
    let bit_cols = product(&[cols, bits.bits() as usize], entry)?;
    // `bit_cols` is a multiple of 32: see the assertion on
    // `DEQUANT_AFFINE_GROUP_SIZES`, so no whole-word check is needed here.
    let words_per_row = bit_cols / 32;
    let words = product(&[rows, words_per_row], entry)?;
    let groups = product(&[rows, cols / shape.group_size as usize], entry)?;
    let elems = product(&[rows, cols], entry)?;
    if words > u32::MAX as usize {
        return Err(format!("{entry}: {words} packed words exceed Metal uint indexing"));
    }
    require::<u32>(rt, bank.packed, words, format_args!("{entry} packed"))?;
    require::<u16>(rt, bank.scales, groups, format_args!("{entry} scales"))?;
    require::<u16>(rt, bank.biases, groups, format_args!("{entry} biases"))?;
    require::<u16>(rt, out, elems, format_args!("{entry} out"))?;
    require::<u32>(rt, fault, 1, format_args!("{entry} fault"))?;
    require_disjoint_writes(
        entry,
        &[("out", out), ("fault", fault)],
        &[
            ("packed", bank.packed),
            ("scales", bank.scales),
            ("biases", bank.biases),
        ],
    )?;
    Ok(Plan {
        words,
        group_shift: shape.group_size.trailing_zeros(),
    })
}

/// Check everything [`dequant_affine_bf16`] checks, without encoding.
pub fn validate_dequant_affine_bf16(
    rt: &GpuRuntime,
    bits: AffineBits,
    bank: DequantAffineBank<'_>,
    out: &GpuBuffer,
    fault: &GpuBuffer,
    shape: QuantShape,
) -> Result<(), String> {
    plan(rt, bits, &bank, out, fault, &shape).map(|_| ())
}

/// Dequantise `bank` (`shape`, `bits`-bit codes) into the bf16 matrix `out`,
/// raising `fault` (one `u32`, see the module docs) if any output element is
/// non-finite.
///
/// Encodes one dispatch into the runtime's current batch; the result is
/// readable after [`GpuRuntime::synchronize`] or by a later GPU pass. Refuses,
/// before encoding anything, an unsupported group size, a `cols` that is not a
/// multiple of the group, size arithmetic that overflows, any buffer shorter
/// than the shape needs (`fault` needs one `u32`) or from another runtime, and
/// an `out` or `fault` that overlaps an input or each other. Zero rows or columns is a no-op once the shape is validated.
///
/// # Fault word protocol
///
/// This function never reads `fault`; reading it is the caller's job, and
/// skipping it is the explicit opt-out. The protocol:
///
/// 1. Zero the word with [`GpuBuffer::try_zero`] (give this run its own word:
///    a faulting thread stores `1` over whatever was there).
/// 2. Dispatch any number of dequants into it.
/// 3. [`GpuRuntime::synchronize`], and treat a sync error as "nothing is
///    known". The synchronize is mandatory.
/// 4. Read the word once with [`check_dequant_affine_fault`]. On `Err`, do not
///    use any `out` written since the zero.
pub fn dequant_affine_bf16(
    rt: &Arc<GpuRuntime>,
    bits: AffineBits,
    bank: DequantAffineBank<'_>,
    out: &GpuBuffer,
    fault: &GpuBuffer,
    shape: QuantShape,
) -> Result<(), String> {
    let p = plan(rt, bits, &bank, out, fault, &shape)?;
    if p.words == 0 {
        return Ok(());
    }
    let pipeline = rt.pipeline(bits.entry())?;
    dispatch_1d(rt, &pipeline, p.words, |bnd| {
        set_gpu_buf(bnd, bank.packed, 0);
        set_gpu_buf(bnd, bank.scales, 1);
        set_gpu_buf(bnd, bank.biases, 2);
        set_gpu_buf(bnd, out, 3);
        set_gpu_buf(bnd, fault, 4);
        set_u32(bnd, p.words as u32, 5);
        set_u32(bnd, p.group_shift, 6);
    })
}

/// Why [`check_dequant_affine_fault`] did not return `Ok`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DequantAffineFault {
    /// The word was read and is non-zero: some output written since the caller
    /// zeroed it holds a non-finite value (or the caller left a non-zero word).
    Raised {
        /// The word as read. A faulting dequant stores `1`; any other value is
        /// one the caller left there.
        word: u32,
    },
    /// The word could not be read, so nothing is known about the outputs.
    Unread {
        /// What stopped the read.
        why: String,
    },
}

impl std::fmt::Display for DequantAffineFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DequantAffineFault::Raised { word } => {
                write!(
                    f,
                    "dequant_affine fault word is {word:#x}: an output holds a non-finite value"
                )
            }
            DequantAffineFault::Unread { why } => {
                write!(f, "dequant_affine fault word was not read: {why}")
            }
        }
    }
}

impl std::error::Error for DequantAffineFault {}

/// The checked read of the fault word `fault` passed to [`dequant_affine_bf16`]:
/// `Ok(())` only when the word was read and is zero.
///
/// A word that is non-zero is [`DequantAffineFault::Raised`]; a buffer that is
/// shorter than a `u32`, or one the host cannot map (a poisoned or busy
/// runtime, private storage, a failed commit), is
/// [`DequantAffineFault::Unread`], never `Ok`. Only the first four bytes are
/// read, and the buffer's length need not be a multiple of four. The word is
/// never written: the fault is sticky, and the caller resets it with
/// [`GpuBuffer::try_zero`].
///
/// This is a host access: mapping the buffer commits the runtime's pending
/// work. Call it at a point where the caller has already called
/// [`GpuRuntime::synchronize`] (see the protocol on [`dequant_affine_bf16`]).
pub fn check_dequant_affine_fault(fault: &GpuBuffer) -> Result<(), DequantAffineFault> {
    if fault.nbytes() < 4 {
        return Err(DequantAffineFault::Unread {
            why: format!("the fault buffer is {} bytes, a u32 needs 4", fault.nbytes()),
        });
    }
    let mapping = fault
        .try_contents_u8()
        .map_err(|why| DequantAffineFault::Unread { why })?;
    let word = u32::from_ne_bytes([mapping[0], mapping[1], mapping[2], mapping[3]]);
    if word == 0 {
        Ok(())
    } else {
        Err(DequantAffineFault::Raised { word })
    }
}

/// [`dequant_affine_bf16`] for 8-bit codes (`dequant_affine_q8_bf16`).
pub fn dequant_affine_q8_bf16(
    rt: &Arc<GpuRuntime>,
    bank: DequantAffineBank<'_>,
    out: &GpuBuffer,
    fault: &GpuBuffer,
    shape: QuantShape,
) -> Result<(), String> {
    dequant_affine_bf16(rt, AffineBits::Q8, bank, out, fault, shape)
}

/// [`dequant_affine_bf16`] for 4-bit codes (`dequant_affine_q4_bf16`).
pub fn dequant_affine_q4_bf16(
    rt: &Arc<GpuRuntime>,
    bank: DequantAffineBank<'_>,
    out: &GpuBuffer,
    fault: &GpuBuffer,
    shape: QuantShape,
) -> Result<(), String> {
    dequant_affine_bf16(rt, AffineBits::Q4, bank, out, fault, shape)
}
