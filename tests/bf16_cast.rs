//! The device `cast_f32_to_bf16` and the host `f32_to_bf16_bits` must agree
//! bit for bit: a training GEMM casts its operands on the device while tests
//! and weight loaders round on the host. The kernel comment used to call the
//! device cast a truncation; both are round-to-nearest-even, and this pins it
//! at the values where truncation and rounding differ.

mod common;

use common::with_gpu;
use tessl::gemm::{cast_bf16_to_f32, cast_bf16_to_f32_into, cast_f32_to_bf16};
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};

#[test]
fn device_bf16_cast_rounds_to_nearest_even_like_the_host() {
    with_gpu(|rt| {
        // Bit patterns around the rounding boundary: the discarded low half is
        // exactly half, just below, just above, with both parities of the kept
        // mantissa so ties-to-even is distinguishable from ties-away, plus the
        // overflow and signed-zero corners. A truncating cast fails all of the
        // "up" cases.
        let patterns: [u32; 10] = [
            0x3F80_8000, // 1.0 + half a bf16 ulp, kept lsb 0: tie → stays
            0x3F81_8000, // kept lsb 1: tie → rounds up to even
            0x3F80_7FFF, // below half: down
            0x3F80_8001, // above half: up
            0xBF80_8001, // negative, above half: away from zero
            0x7F7F_FFFF, // f32::MAX rounds up to +inf in bf16
            0x4B7F_FFFF, // 2^24 - 1: mantissa all ones, carries into the exponent
            0x0000_0001, // smallest subnormal rounds to +0
            0x8000_0000, // -0 stays -0
            0x3F7F_FFFF, // just below 1.0: rounds up to exactly 1.0
        ];
        let values: Vec<f32> = patterns.iter().map(|p| f32::from_bits(*p)).collect();
        let src = rt.alloc_tensor_f32(&[values.len()]).unwrap();
        src.buffer.write_f32(&values);

        let bf16 = cast_f32_to_bf16(&src).unwrap();
        let back = cast_bf16_to_f32(&bf16).unwrap();
        rt.synchronize().unwrap();
        let got = back.buffer.read_f32();

        for (i, (value, g)) in values.iter().zip(&got).enumerate() {
            let want = bf16_bits_to_f32(f32_to_bf16_bits(*value));
            assert_eq!(
                g.to_bits(),
                want.to_bits(),
                "pattern {:#010x} ({value:e}): device gave {g:e} ({:#06x}), host {want:e}",
                patterns[i],
                f32_to_bf16_bits(*g)
            );
        }
    });
}

/// The in-place widening writes only its destination view, exactly, and
/// refuses a destination it cannot fill: wrong dtype, wrong shape, or one
/// overlapping the source.
#[test]
fn bf16_widening_into_a_view_is_exact_and_checks_its_destination() {
    with_gpu(|rt| {
        let values = [1.5f32, -0.0, 3.0e38, -7.25, 1.0e-40, 65504.0];
        let src = rt.alloc_tensor_bf16(&[2, 3]).unwrap();
        let bits: Vec<u16> = values.iter().map(|v| f32_to_bf16_bits(*v)).collect();
        src.buffer.write_bf16_bits(&bits);

        // A [2, 3] view at element 4 of a 12-element buffer: 4 before, 2 after.
        let big = rt.alloc_tensor_f32(&[12]).unwrap();
        big.write_f32(&[99.0; 12]).unwrap();
        let dst = big.try_view(&[2, 3], 4).unwrap();
        cast_bf16_to_f32_into(&src, &dst).unwrap();
        rt.synchronize().unwrap();
        let got = big.read_f32().unwrap();
        assert_eq!(&got[..4], &[99.0; 4], "wrote before the view");
        assert_eq!(&got[10..], &[99.0; 2], "wrote past the view");
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(got[4 + i].to_bits(), bf16_bits_to_f32(b).to_bits(), "element {i}");
        }

        let reject = |dst: &tessl::tensor::Tensor, what: &str| {
            let e = cast_bf16_to_f32_into(&src, dst).expect_err(what);
            assert!(e.contains("cast destination"), "{what}: {e}");
        };
        reject(&big.try_view(&[3, 2], 0).unwrap(), "transposed shape");
        reject(&big.try_view(&[6], 0).unwrap(), "flattened shape");
        reject(&rt.alloc_tensor_bf16(&[2, 3]).unwrap(), "bf16 destination");
        // The source's own bytes viewed as f32 (12 bytes = 3 floats).
        let alias = tessl::tensor::Tensor::from_buffer(
            rt,
            src.buffer.clone(),
            &[1, 3],
            tessl::tensor::DType::F32,
            0,
        )
        .unwrap();
        let e = cast_bf16_to_f32_into(&src, &alias).expect_err("aliasing destination");
        assert!(e.contains("cast destination"), "aliasing: {e}");
    });
}
