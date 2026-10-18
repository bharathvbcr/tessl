//! The inverse DCT of libjpeg-turbo's accurate integer method
//! (`jpeg_idct_islow`, `JDCT_ISLOW`) as its 64-bit ARM NEON implementation
//! computes it, which is what the reference runs on this machine.
//!
//! For every block whose dequantized coefficients stay inside 16 bits (all valid
//! files) this is the same arithmetic as `jidctint.c`: the same 13-bit
//! constants, the same two passes, the same rounding. They differ only on
//! corrupt data, where the C code works in 64 bits and wraps the final value
//! with a 1024-entry range table, while the NEON code works in 16-bit lanes
//! (dequantized values, and the sums that feed the even and odd parts, wrap at
//! 16 bits), accumulates in 32 bits (wrapping) and saturates the sample. The
//! reference's output on such blocks is the NEON one, and this reproduces it:
//! checked against torchvision on dense random blocks with coefficients of
//! +-32767, quantizers up to 65535, and on DC-only sweeps.
//!
//! The transform is written once as an 8-point routine applied to columns, then
//! to rows. A column with no AC coefficient takes the same value through the
//! general path as through the DC shortcut (`dc << 2`), so no special case is
//! needed for equality.

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

const F_0_298: i32 = 2446;
const F_0_390: i32 = 3196;
const F_0_541: i32 = 4433;
const F_0_765: i32 = 6270;
const F_0_899: i32 = 7373;
const F_1_175: i32 = 9633;
const F_1_501: i32 = 12299;
const F_1_847: i32 = 15137;
const F_1_961: i32 = 16069;
const F_2_053: i32 = 16819;
const F_2_562: i32 = 20995;
const F_3_072: i32 = 25172;

/// `a * b + c * d` in 32-bit wrapping arithmetic (`vmull` then `vmlal`).
#[inline(always)]
fn mac(a: i16, b: i32, c: i16, d: i32) -> i32 {
    i32::from(a).wrapping_mul(b).wrapping_add(i32::from(c).wrapping_mul(d))
}

/// One 8-point inverse transform of 16-bit inputs `d[0..8]` (frequencies in
/// order), giving the eight outputs before descaling.
#[inline(always)]
fn transform(d: &[i16; 8]) -> [i32; 8] {
    // Even part.
    let tmp2 = mac(d[2], F_0_541, d[6], F_0_541 - F_1_847);
    let tmp3 = mac(d[2], F_0_541 + F_0_765, d[6], F_0_541);
    let tmp0 = i32::from(d[0].wrapping_add(d[4])) << CONST_BITS;
    let tmp1 = i32::from(d[0].wrapping_sub(d[4])) << CONST_BITS;
    let tmp10 = tmp0.wrapping_add(tmp3);
    let tmp13 = tmp0.wrapping_sub(tmp3);
    let tmp11 = tmp1.wrapping_add(tmp2);
    let tmp12 = tmp1.wrapping_sub(tmp2);
    // Odd part, from the inputs at frequencies 7, 5, 3, 1.
    let (t0, t1, t2, t3) = (d[7], d[5], d[3], d[1]);
    let z3s = t0.wrapping_add(t2);
    let z4s = t1.wrapping_add(t3);
    let z3 = mac(z3s, F_1_175 - F_1_961, z4s, F_1_175);
    let z4 = mac(z3s, F_1_175, z4s, F_1_175 - F_0_390);
    let o0 = mac(t0, F_0_298 - F_0_899, t3, -F_0_899).wrapping_add(z3);
    let o1 = mac(t1, F_2_053 - F_2_562, t2, -F_2_562).wrapping_add(z4);
    let o2 = mac(t2, F_3_072 - F_2_562, t1, -F_2_562).wrapping_add(z3);
    let o3 = mac(t3, F_1_501 - F_0_899, t0, -F_0_899).wrapping_add(z4);
    [
        tmp10.wrapping_add(o3),
        tmp11.wrapping_add(o2),
        tmp12.wrapping_add(o1),
        tmp13.wrapping_add(o0),
        tmp13.wrapping_sub(o0),
        tmp12.wrapping_sub(o1),
        tmp11.wrapping_sub(o2),
        tmp10.wrapping_sub(o3),
    ]
}

/// Dequantize and inverse-transform one block, writing 8 rows of 8 samples at
/// `out[row * stride ..]`.
pub(crate) fn islow(coef: &[i16; 64], quant: &[u16; 64], out: &mut [u8], stride: usize) {
    // `vmul_s16` on the coefficients and the 16-bit quantizer table.
    let mut dq = [0i16; 64];
    for i in 0..64 {
        dq[i] = coef[i].wrapping_mul(quant[i] as i16);
    }
    let mut ws = [0i16; 64];
    for col in 0..8 {
        let d = [
            dq[col],
            dq[8 + col],
            dq[16 + col],
            dq[24 + col],
            dq[32 + col],
            dq[40 + col],
            dq[48 + col],
            dq[56 + col],
        ];
        if (1..8).all(|r| coef[r * 8 + col] == 0) {
            let v = d[0].wrapping_shl(PASS1_BITS as u32);
            for r in 0..8 {
                ws[r * 8 + col] = v;
            }
            continue;
        }
        let sums = transform(&d);
        for r in 0..8 {
            // `vrshrn_n_s32`: round, shift, truncate to 16 bits.
            ws[r * 8 + col] =
                ((i64::from(sums[r]) + (1 << (CONST_BITS - PASS1_BITS - 1))) >> (CONST_BITS - PASS1_BITS)) as i16;
        }
    }
    for row in 0..8 {
        let w: [i16; 8] = ws[row * 8..row * 8 + 8].try_into().expect("8 values");
        let sums = transform(&w);
        let o = &mut out[row * stride..row * stride + 8];
        for c in 0..8 {
            // Descale by 18 bits in two steps (>> 16 truncating, then a rounding
            // >> 2), add the sample centre and saturate.
            let y = sums[c] >> 16;
            o[c] = (((y + 2) >> 2) + 128).clamp(0, 255) as u8;
        }
    }
}
