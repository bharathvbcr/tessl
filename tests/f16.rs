//! IEEE binary16 support: host conversion, device casts, and GEMM.
//!
//! f16 and bf16 are both two bytes, both accumulate in f32 here, and are *not*
//! interchangeable: bf16 shares f32's exponent field, f16 does not. A buffer
//! written as one and read as the other is silently wrong rather than
//! imprecise, so the tests below check the bit layout directly rather than
//! only checking that numbers come out roughly right.

mod common;

use std::sync::Arc;

use common::{random_f32, with_gpu};
use tessl::gemm::{cast_f16_to_f32, cast_f32_to_f16, gemm, GemmBackend};
use tessl::tensor::{f16_bits_to_f32, f32_slice_to_f16, f32_to_f16_bits, DType, Tensor};
use tessl::GpuRuntime;

fn f16_tensor(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f16(shape).expect("alloc_tensor_f16");
    t.buffer.write_f16_bits(&f32_slice_to_f16(data));
    t
}

#[test]
fn host_conversion_round_trips_and_matches_known_bit_patterns() {
    // Anchors from IEEE 754 binary16, so this checks the encoding rather than
    // checking the function against itself.
    let cases: &[(f32, u16)] = &[
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (1.0, 0x3c00),
        (-2.0, 0xc000),
        (0.5, 0x3800),
        (65504.0, 0x7bff),            // largest finite half
        (1.0 / 16384.0, 0x0400),      // 2^-14, the smallest normal
        (1.0 / 16_777_216.0, 0x0001), // 2^-24, the smallest subnormal
        (f32::INFINITY, 0x7c00),
    ];
    for &(v, bits) in cases {
        assert_eq!(f32_to_f16_bits(v), bits, "encoding {v}");
        assert_eq!(
            f16_bits_to_f32(bits).to_bits(),
            v.to_bits(),
            "decoding {v:?}"
        );
    }

    // Overflow saturates to infinity. This is the difference from bf16 that
    // matters: bf16 has f32's exponent range and would keep the value.
    assert_eq!(f32_to_f16_bits(70000.0), 0x7c00);
    assert_eq!(f32_to_f16_bits(-70000.0), 0xfc00);
    assert_eq!(f32_to_f16_bits(3.4e38), 0x7c00);
    // Underflow past subnormal is a signed zero, not a denormal-of-a-denormal.
    assert_eq!(f32_to_f16_bits(1e-10), 0x0000);
    assert_eq!(f32_to_f16_bits(-1e-10), 0x8000);
    // NaN stays NaN even though every payload bit is dropped.
    assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
}

#[test]
fn host_round_trip_is_exact_for_representable_values() {
    // Every half is exactly representable in f32, so decode-then-encode must
    // be the identity on all 65536 bit patterns.
    for bits in 0u32..=0xffff {
        let b = bits as u16;
        let v = f16_bits_to_f32(b);
        if v.is_nan() {
            continue; // NaN payloads are not required to survive
        }
        assert_eq!(f32_to_f16_bits(v), b, "round trip of bit pattern {b:#06x}");
    }
}

#[test]
fn device_casts_agree_with_the_host_conversion() {
    with_gpu(|rt| {
        let n = 4096usize;
        // Spread across normals, subnormals and the overflow edge.
        let mut data = random_f32(n, 0xF16);
        for (i, v) in data.iter_mut().enumerate() {
            *v *= match i % 4 {
                0 => 1.0,
                1 => 1e-6,
                2 => 1e4,
                _ => 1e5,
            };
        }
        let src = rt.alloc_tensor_f32(&[n]).unwrap();
        src.buffer.write_f32(&data);

        let half = cast_f32_to_f16(&src).unwrap();
        assert_eq!(half.dtype, DType::F16);
        let back = cast_f16_to_f32(&half).unwrap();
        rt.synchronize().unwrap();

        let got = back.buffer.read_f32();
        for (i, v) in data.iter().enumerate() {
            let want = f16_bits_to_f32(f32_to_f16_bits(*v));
            assert_eq!(
                got[i].to_bits(),
                want.to_bits(),
                "device cast disagrees with the host at {i}: input {v}"
            );
        }
    });
}

#[test]
fn f16_and_bf16_are_not_interchangeable() {
    // The one-line summary of why this is its own dtype. If a caller could
    // pass f16 bits to a bf16 kernel and get plausible numbers, the mistake
    // would never be caught.
    use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};
    let v = 1.0f32;
    assert_ne!(
        f32_to_f16_bits(v),
        f32_to_bf16_bits(v),
        "1.0 must encode differently in f16 and bf16"
    );
    // Reading f16 bits as bf16 gives a wildly different number, not a nearby
    // one — which is what makes a silent mix-up so damaging.
    let as_bf16 = bf16_bits_to_f32(f32_to_f16_bits(v));
    assert!(
        (as_bf16 - v).abs() > 0.5,
        "f16 bits read as bf16 landed suspiciously close to the original"
    );
}

#[test]
fn f16_gemm_matches_an_f32_reference_within_f16_resolution() {
    with_gpu(|rt| {
        for &(m, n, k) in &[(128usize, 64usize, 64usize), (96, 80, 128)] {
            let a_h = random_f32(m * k, 0x1F16);
            let b_h = random_f32(k * n, 0x2F16);
            let a = f16_tensor(rt, &[m, k], &a_h);
            let b = f16_tensor(rt, &[k, n], &b_h);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            c.buffer.write_f32(&vec![0.0f32; m * n]);

            gemm(&a, &b, &c, GemmBackend::TensorOps).expect("f16 gemm");
            rt.synchronize().unwrap();

            // Reference over the *rounded* operands: the kernel never sees the
            // originals, so comparing against them would measure the host cast.
            let ar: Vec<f32> = a_h
                .iter()
                .map(|v| f16_bits_to_f32(f32_to_f16_bits(*v)))
                .collect();
            let br: Vec<f32> = b_h
                .iter()
                .map(|v| f16_bits_to_f32(f32_to_f16_bits(*v)))
                .collect();
            let got = c.buffer.read_f32();
            let mut worst = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for p in 0..k {
                        acc += ar[i * k + p] as f64 * br[p * n + j] as f64;
                    }
                    worst = worst.max((got[i * n + j] as f64 - acc).abs() as f32);
                }
            }
            // f16 has 10 mantissa bits; the products are exact in the f32
            // accumulator, so the error is the accumulation order only.
            let bound = 8.0 * f32::EPSILON * k as f32;
            assert!(
                worst <= bound.max(1e-3),
                "{m}x{n}x{k}: worst |delta| = {worst}, bound {bound}"
            );
        }
    });
}

#[test]
fn mixing_f16_with_another_dtype_is_refused() {
    with_gpu(|rt| {
        let (m, n, k) = (128usize, 64usize, 64usize);
        let a = f16_tensor(rt, &[m, k], &vec![1.0f32; m * k]);
        let b = rt.alloc_tensor_bf16(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        let err = gemm(&a, &b, &c, GemmBackend::TensorOps).expect_err("mixed dtypes");
        assert!(err.contains("matching operand dtypes"), "{err}");

        // And f16 on a backend with no f16 kernel.
        let b16 = f16_tensor(rt, &[k, n], &vec![1.0f32; k * n]);
        let err = gemm(&a, &b16, &c, GemmBackend::Simdgroup).expect_err("simdgroup f16");
        assert!(err.contains("TensorOps"), "{err}");
    });
}
