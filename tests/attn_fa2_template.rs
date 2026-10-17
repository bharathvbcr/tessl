//! Bit-exact pin on the three tiled FA-2 kernels (`flash_attn_swa_h128`,
//! `flash_attn_swa_h256`, `flash_attn_global_h512`).
//!
//! The three share one body (`kernels/attn_fa2.h`). This test runs each on fixed
//! seeded inputs, including rows with no live key, edge sizes, an empty history
//! and the h512 bf16 output, and compares an FNV-1a hash of the raw output words
//! with a value recorded from the kernels *before* they were folded into the
//! template. Any change to the arithmetic or its order moves a hash, which a
//! tolerance-based comparison against the f64 reference in `attention.rs` would
//! not notice.

mod common;

use common::{buf, random_f32, seeded, with_gpu};
use std::sync::Arc;
use tessl::nn::{self, AttnDims, AttnHeadDim};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

struct Case {
    name: &'static str,
    b: usize,
    tq: usize,
    tkv: usize,
    cap: usize,
    h: usize,
    hkv: usize,
    window: u32,
    q_off: u32,
    kv_off: u32,
    bf16_out: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "prefill-window5",
        b: 2,
        tq: 11,
        tkv: 11,
        cap: 11,
        h: 4,
        hkv: 2,
        window: 5,
        q_off: 0,
        kv_off: 0,
        bf16_out: false,
    },
    Case {
        name: "rows-without-keys",
        b: 1,
        tq: 9,
        tkv: 9,
        cap: 9,
        h: 2,
        hkv: 1,
        window: 4,
        q_off: 0,
        kv_off: 3,
        bf16_out: false,
    },
    Case {
        name: "rows-without-keys-bf16",
        b: 1,
        tq: 9,
        tkv: 9,
        cap: 9,
        h: 2,
        hkv: 1,
        window: 4,
        q_off: 0,
        kv_off: 3,
        bf16_out: true,
    },
    Case {
        name: "decode-window8",
        b: 1,
        tq: 1,
        tkv: 37,
        cap: 40,
        h: 4,
        hkv: 2,
        window: 8,
        q_off: 36,
        kv_off: 0,
        bf16_out: false,
    },
    Case {
        name: "empty-history",
        b: 2,
        tq: 9,
        tkv: 0,
        cap: 3,
        h: 4,
        hkv: 2,
        window: 0,
        q_off: 17,
        kv_off: 41,
        bf16_out: false,
    },
    Case {
        name: "unbounded-ragged",
        b: 1,
        tq: 3,
        tkv: 13,
        cap: 13,
        h: 2,
        hkv: 2,
        window: 0,
        q_off: 20,
        kv_off: 10,
        bf16_out: false,
    },
    Case {
        name: "unbounded-ragged-bf16",
        b: 1,
        tq: 3,
        tkv: 13,
        cap: 13,
        h: 2,
        hkv: 2,
        window: 0,
        q_off: 20,
        kv_off: 10,
        bf16_out: true,
    },
    Case {
        name: "window1-prefill",
        b: 1,
        tq: 17,
        tkv: 17,
        cap: 17,
        h: 2,
        hkv: 1,
        window: 1,
        q_off: 0,
        kv_off: 0,
        bf16_out: false,
    },
];

fn u32_buf(rt: &Arc<GpuRuntime>, v: u32) -> GpuBuffer {
    let b = rt.alloc_buffer(4).unwrap();
    b.write_u32(&[v]);
    b
}

fn fnv1a(words: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for byte in w.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Hash of the raw output words of one kernel on one case. `None` when the
/// kernel has no bf16 output for the case.
fn run(rt: &Arc<GpuRuntime>, d: usize, c: &Case) -> Option<u64> {
    if c.bf16_out && d != 512 {
        return None;
    }
    let n = c.b * c.tq * c.h * d;
    let kv = c.b * c.cap * c.hkv * d;
    let seed = 0xfa20 + d as u64;
    let q = buf(rt, &random_f32(n, seed));
    let k = buf(rt, &random_f32(kv, seed + 1));
    let v = buf(rt, &random_f32(kv, seed + 2));
    // Twice the f32 footprint, filled with a sentinel, so unwritten words (and
    // the bf16 half-width layout) are part of what is hashed.
    let o = seeded(rt, n, f32::from_bits(0xA5A5_A5A5));
    let tkv = u32_buf(rt, c.tkv as u32);
    let qo = u32_buf(rt, c.q_off);
    let ko = u32_buf(rt, c.kv_off);
    let dims = AttnDims {
        batch: c.b as u32,
        tq: c.tq as u32,
        heads: c.h as u32,
        heads_kv: c.hkv as u32,
        window: c.window,
        scale: 0.125,
    };
    match d {
        128 => nn::flash_attn_swa_tiled(rt, AttnHeadDim::D128, &q, &k, &v, &o, &tkv, &qo, &ko, dims).unwrap(),
        256 => nn::flash_attn_swa_tiled(rt, AttnHeadDim::D256, &q, &k, &v, &o, &tkv, &qo, &ko, dims).unwrap(),
        512 => nn::flash_attn_global_h512_tiled(rt, &q, &k, &v, &o, &tkv, &qo, &ko, dims, c.bf16_out).unwrap(),
        _ => unreachable!(),
    }
    rt.synchronize().unwrap();
    Some(fnv1a(&o.read_u32()[..n]))
}

/// `(kernel head dim, case name, hash)` recorded from the pre-template kernels.
const GOLDEN: &[(usize, &str, u64)] = &[
    (128, "prefill-window5", 0xe5afd73bfb912640),
    (128, "rows-without-keys", 0x64ea51b5dc9c9e90),
    (128, "decode-window8", 0xe7b5afc1ef849939),
    (128, "empty-history", 0xbe7134315f056325),
    (128, "unbounded-ragged", 0x5c818bc52942ac5f),
    (128, "window1-prefill", 0x628ae11e33d62c7d),
    (256, "prefill-window5", 0x85d3b80e69ce3836),
    (256, "rows-without-keys", 0x3b7aa2c57b466e03),
    (256, "decode-window8", 0xe9b189e2a74f1cfc),
    (256, "empty-history", 0x0668679189e8a325),
    (256, "unbounded-ragged", 0x1abc626af83d736a),
    (256, "window1-prefill", 0xb1f1e6415e87352d),
    (512, "prefill-window5", 0x9c6f51e4c3c69c11),
    (512, "rows-without-keys", 0x0ee3a7ef04eeaf19),
    (512, "rows-without-keys-bf16", 0x9141e45635d7ae2e),
    (512, "decode-window8", 0xf8a13e4a1518cf74),
    (512, "empty-history", 0x01a2728bcfaf2325),
    (512, "unbounded-ragged", 0x33cfb11621287e01),
    (512, "unbounded-ragged-bf16", 0x901ab2728d83ccc0),
    (512, "window1-prefill", 0x609ceb6a879086ab),
];

#[test]
fn tiled_fa2_kernels_are_bit_identical_to_the_recorded_pre_template_output() {
    with_gpu(|rt| {
        let mut got = Vec::new();
        for d in [128usize, 256, 512] {
            for c in CASES {
                if let Some(h) = run(rt, d, c) {
                    got.push((d, c.name, h));
                }
            }
        }
        for (d, name, h) in &got {
            eprintln!("GOLDEN ({d}, \"{name}\", {h:#018x}),");
        }
        assert_eq!(got, GOLDEN, "tiled FA-2 output bits moved");
    });
}
