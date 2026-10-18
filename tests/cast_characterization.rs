//! Characterization of the four device casts, recorded on the code before the
//! `_into` casts were routed through the shared `cast_between` dispatch. The
//! outputs are pinned as FNV-1a digests of the exact result bits over the full
//! 16-bit pattern space (NaN payloads, infinities, signed zeros, subnormals,
//! ties and near-ties), so any change in a cast's bit behaviour fails here
//! instead of passing a tolerance.

mod common;

use common::with_gpu;
use std::sync::Arc;
use tessl::gemm::{
    cast_bf16_to_f32, cast_bf16_to_f32_into, cast_f16_to_f32, cast_f32_to_bf16, cast_f32_to_bf16_into, cast_f32_to_f16,
};
use tessl::GpuRuntime;

fn fnv(bits: impl IntoIterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bits {
        for byte in b.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Every high half crossed with low halves around the rounding boundary:
/// exact tie, one below/above it, zero, one, all ones. Covers every NaN
/// payload class reachable from the high half, both infinities, both zeros,
/// subnormals and every kept-lsb parity.
fn f32_patterns() -> Vec<f32> {
    let lows = [0x0000u32, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF];
    let mut v = Vec::with_capacity(65536 * lows.len());
    for hi in 0..=0xFFFFu32 {
        for lo in lows {
            v.push(f32::from_bits((hi << 16) | lo));
        }
    }
    v
}

fn f32_src(rt: &Arc<GpuRuntime>, data: &[f32]) -> tessl::tensor::Tensor {
    let t = rt.alloc_tensor_f32(&[data.len()]).unwrap();
    t.buffer.write_f32(data);
    t
}

fn digest_f32(rt: &Arc<GpuRuntime>, t: &tessl::tensor::Tensor) -> u64 {
    rt.synchronize().unwrap();
    fnv(t.buffer.read_f32().iter().map(|x| x.to_bits()))
}

const F32_TO_BF16: u64 = 0x705060939f3660a5;
const F32_TO_BF16_INTO: u64 = 0x705060939f3660a5;
const BF16_TO_F32: u64 = 0x43a13836c4ca3b25;
const BF16_TO_F32_INTO: u64 = 0x43a13836c4ca3b25;
const F32_TO_F16: u64 = 0xd0dcdfa9e11e7fc5;
const F16_TO_F32: u64 = 0x1f87f165810ea7c5;

#[test]
fn casts_are_bit_stable_over_the_pattern_space() {
    with_gpu(|rt| {
        let data = f32_patterns();
        let src = f32_src(rt, &data);

        let bf = cast_f32_to_bf16(&src).unwrap();
        let bf_back = cast_bf16_to_f32(&bf).unwrap();
        let d_bf = digest_f32(rt, &bf_back);

        let bf_into = rt.alloc_tensor_bf16(&[data.len()]).unwrap();
        cast_f32_to_bf16_into(&src, &bf_into).unwrap();
        let wide_into = rt.alloc_tensor_f32(&[data.len()]).unwrap();
        cast_bf16_to_f32_into(&bf_into, &wide_into).unwrap();
        let d_bf_into = digest_f32(rt, &wide_into);

        let h = cast_f32_to_f16(&src).unwrap();
        let h_back = cast_f16_to_f32(&h).unwrap();
        let d_h = digest_f32(rt, &h_back);

        // Widening from every 16-bit pattern, as bf16 and as f16.
        let all: Vec<u16> = (0..=0xFFFFu16).collect();
        let bsrc = rt.alloc_tensor_bf16(&[all.len()]).unwrap();
        bsrc.buffer.write_bf16_bits(&all);
        let bw = cast_bf16_to_f32(&bsrc).unwrap();
        let d_bw = digest_f32(rt, &bw);
        let bw_into = rt.alloc_tensor_f32(&[all.len()]).unwrap();
        cast_bf16_to_f32_into(&bsrc, &bw_into).unwrap();
        let d_bw_into = digest_f32(rt, &bw_into);
        let hsrc = rt.alloc_tensor_f16(&[all.len()]).unwrap();
        hsrc.buffer.write_f16_bits(&all);
        let hw = cast_f16_to_f32(&hsrc).unwrap();
        let d_hw = digest_f32(rt, &hw);

        eprintln!(
            "DIGESTS bf={d_bf:#x} bf_into={d_bf_into:#x} h={d_h:#x} bw={d_bw:#x} bw_into={d_bw_into:#x} hw={d_hw:#x}"
        );
        assert_eq!(d_bf, F32_TO_BF16, "f32->bf16 (alloc)");
        assert_eq!(d_bf_into, F32_TO_BF16_INTO, "f32->bf16 (_into)");
        assert_eq!(d_bf, d_bf_into, "alloc and _into narrowing agree");
        assert_eq!(d_h, F32_TO_F16, "f32->f16");
        assert_eq!(d_bw, BF16_TO_F32, "bf16->f32 (alloc), all 65536 patterns");
        assert_eq!(d_bw_into, BF16_TO_F32_INTO, "bf16->f32 (_into), all 65536 patterns");
        assert_eq!(d_hw, F16_TO_F32, "f16->f32, all 65536 patterns");
    });
}
