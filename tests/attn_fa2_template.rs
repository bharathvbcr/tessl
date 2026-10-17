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

/// Raw output words (`n` of them) of one kernel on one case, with the device
/// `Tkv` word set to `dev_tkv`. `None` when the kernel has no bf16 output for
/// the case.
fn run_words(rt: &Arc<GpuRuntime>, d: usize, c: &Case, dev_tkv: u32) -> Option<Vec<u32>> {
    if c.bf16_out && d != 512 {
        return None;
    }
    let n = c.b * c.tq * c.h * d;
    let kv = c.b * c.cap * c.hkv * d;
    let seed = 0xfa20 + d as u64;
    let q = buf(rt, &random_f32(n, seed));
    let k = buf(rt, &random_f32(kv, seed + 1));
    let v = buf(rt, &random_f32(kv, seed + 2));
    // `n` words (the f32 footprint), filled with a sentinel and all read back,
    // so a word the kernel never wrote is part of what is compared. A bf16
    // output fills only the first half; the rest keeps the sentinel.
    let o = seeded(rt, n, f32::from_bits(0xA5A5_A5A5));
    let tkv = u32_buf(rt, dev_tkv);
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
    Some(o.read_u32()[..n].to_vec())
}

const SAMPLE: usize = 8;

/// `(kernel head dim, case name, hash of every word, first SAMPLE words)`
/// recorded from the pre-template kernels. The samples exist only so a
/// mismatch can name the words that moved; the hash is what decides.
type Golden = (usize, &'static str, u64, [u32; SAMPLE]);

fn sample(words: &[u32]) -> [u32; SAMPLE] {
    let mut s = [0u32; SAMPLE];
    for (dst, src) in s.iter_mut().zip(words) {
        *dst = *src;
    }
    s
}

const GOLDEN: &[Golden] = &[
    (
        128,
        "prefill-window5",
        0xe5afd73bfb912640,
        [
            0xbe743280, 0x3f451396, 0x3e06be40, 0x3ea78070, 0x3f3593dc, 0xbe13b6b0, 0xbf4c67e4, 0xbee7755c,
        ],
    ),
    (128, "rows-without-keys", 0x64ea51b5dc9c9e90, [0; SAMPLE]),
    (
        128,
        "decode-window8",
        0xe7b5afc1ef849939,
        [
            0xbe8cffe4, 0x3eb798b2, 0x3e291364, 0xbc9f3fd3, 0xbec31869, 0x3ad400ad, 0xbefaa9f2, 0x3e6b0adf,
        ],
    ),
    (128, "empty-history", 0xbe7134315f056325, [0; SAMPLE]),
    (
        128,
        "unbounded-ragged",
        0x5c818bc52942ac5f,
        [
            0x3c0e029f, 0x3e446025, 0x3cc5043a, 0x3b06c1a2, 0x3c6f8262, 0xbd128edb, 0xbe38dd0d, 0x3e141468,
        ],
    ),
    (
        128,
        "window1-prefill",
        0x628ae11e33d62c7d,
        [
            0xbe743280, 0x3f451396, 0x3e06be40, 0x3ea78070, 0x3f3593dc, 0xbe13b6b0, 0xbf4c67e4, 0xbee7755c,
        ],
    ),
    (
        256,
        "prefill-window5",
        0x85d3b80e69ce3836,
        [
            0xbf101e48, 0xbe67bd50, 0xbe810a40, 0xbe1b5148, 0x3ee3fa00, 0x3f45599a, 0x3f264284, 0x3f69acfc,
        ],
    ),
    (256, "rows-without-keys", 0x3b7aa2c57b466e03, [0; SAMPLE]),
    (
        256,
        "decode-window8",
        0xe9b189e2a74f1cfc,
        [
            0xbe91c4d2, 0xbdea0c94, 0x3ec86067, 0xbec08b7a, 0x3edbbb6d, 0x3ea3ee9d, 0xbe699ec1, 0x3e7b55c2,
        ],
    ),
    (256, "empty-history", 0x0668679189e8a325, [0; SAMPLE]),
    (
        256,
        "unbounded-ragged",
        0x1abc626af83d736a,
        [
            0x3e92c0fa, 0x3da5a9c0, 0x3ea7f5be, 0x3e98c7f1, 0x3edeea26, 0x3d466c86, 0x3bb67c96, 0xbe66ca5a,
        ],
    ),
    (
        256,
        "window1-prefill",
        0xb1f1e6415e87352d,
        [
            0xbf101e48, 0xbe67bd50, 0xbe810a40, 0xbe1b5148, 0x3ee3fa00, 0x3f45599a, 0x3f264284, 0x3f69acfc,
        ],
    ),
    (
        512,
        "prefill-window5",
        0x9c6f51e4c3c69c11,
        [
            0xbf016e9a, 0x3e1a7b18, 0x3f5ad3f0, 0xbe33d890, 0x3f389dd0, 0xbf61dfea, 0xbe975228, 0x3e584ef0,
        ],
    ),
    (512, "rows-without-keys", 0x0ee3a7ef04eeaf19, [0; SAMPLE]),
    (512, "rows-without-keys-bf16", 0x9141e45635d7ae2e, [0; SAMPLE]),
    (
        512,
        "decode-window8",
        0xf8a13e4a1518cf74,
        [
            0x3ec57726, 0x3c730192, 0x3e759521, 0x3d59d407, 0x3dc80682, 0xb9efee7e, 0xbd32ce38, 0xbd9da96f,
        ],
    ),
    (512, "empty-history", 0x01a2728bcfaf2325, [0; SAMPLE]),
    (
        512,
        "unbounded-ragged",
        0x33cfb11621287e01,
        [
            0x3efab29f, 0xbe3e4f5e, 0x3e8c9e35, 0x3e3cd0b7, 0xbdc53368, 0xbead1ceb, 0xbe322adc, 0xbe0e1d4a,
        ],
    ),
    (
        512,
        "unbounded-ragged-bf16",
        0x901ab2728d83ccc0,
        [
            0xbe3e3efb, 0x3e3d3e8d, 0xbeadbdc5, 0xbe0ebe32, 0xbe2e3e06, 0x3dff3bae, 0xbd89be89, 0xbdb9bdef,
        ],
    ),
    (
        512,
        "window1-prefill",
        0x609ceb6a879086ab,
        [
            0xbf016e9a, 0x3e1a7b18, 0x3f5ad3f0, 0xbe33d890, 0x3f389dd0, 0xbf61dfea, 0xbe975228, 0x3e584ef0,
        ],
    ),
];

#[test]
fn tiled_fa2_kernels_are_bit_identical_to_the_recorded_pre_template_output() {
    with_gpu(|rt| {
        let mut got: Vec<Golden> = Vec::new();
        for d in [128usize, 256, 512] {
            for c in CASES {
                if let Some(w) = run_words(rt, d, c, c.tkv as u32) {
                    got.push((d, c.name, fnv1a(&w), sample(&w)));
                }
            }
        }
        for (d, name, h, s) in &got {
            let words: Vec<String> = s.iter().map(|w| format!("{w:#010x}")).collect();
            eprintln!("    ({d}, \"{name}\", {h:#018x}, [{}]),", words.join(", "));
        }
        let mut moved = Vec::new();
        for (g, w) in got.iter().zip(GOLDEN) {
            if g.2 != w.2 {
                let words: Vec<String> = (0..SAMPLE)
                    .filter(|&i| g.3[i] != w.3[i])
                    .map(|i| format!("[{i}] got {:#010x} want {:#010x}", g.3[i], w.3[i]))
                    .collect();
                let detail = if words.is_empty() {
                    format!("first {SAMPLE} words equal; the change is later in the output")
                } else {
                    words.join(", ")
                };
                moved.push(format!(
                    "h{} {}: hash {:#018x} != {:#018x}; {detail}",
                    g.0, g.1, g.2, w.2
                ));
            }
        }
        assert!(moved.is_empty(), "tiled FA-2 output bits moved:\n{}", moved.join("\n"));
        assert_eq!(got.len(), GOLDEN.len(), "case list and golden table disagree");
        let keys: Vec<_> = got.iter().map(|g| (g.0, g.1)).collect();
        let want_keys: Vec<_> = GOLDEN.iter().map(|g| (g.0, g.1)).collect();
        assert_eq!(keys, want_keys, "case list and golden table disagree");
    });
}

/// A device `Tkv` past `kv_capacity` is clamped to it. The queries sit far past
/// every key so that, unclamped, the keys beyond the buffers would be live and
/// change the answer (or fault); the run with the oversized value must be
/// bit-equal to the run with `Tkv == kv_capacity`.
#[test]
fn device_tkv_above_capacity_is_clamped_in_every_tiled_kernel() {
    with_gpu(|rt| {
        for bf16_out in [false, true] {
            let c = Case {
                name: "clamp",
                b: 2,
                tq: 3,
                tkv: 11,
                cap: 11,
                h: 2,
                hkv: 1,
                window: 0,
                q_off: 100,
                kv_off: 0,
                bf16_out,
            };
            for d in [128usize, 256, 512] {
                let Some(clamped) = run_words(rt, d, &c, c.cap as u32) else {
                    continue;
                };
                // The reference run must have written something.
                assert!(clamped.iter().any(|&w| w != 0xA5A5_A5A5), "d={d}: kernel wrote nothing");
                for over in [c.cap as u32 + 1, c.cap as u32 + 14, u32::MAX] {
                    let got = run_words(rt, d, &c, over).unwrap();
                    let diff: Vec<String> = got
                        .iter()
                        .zip(&clamped)
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .take(4)
                        .map(|(i, (a, b))| format!("[{i}] got {a:#010x} want {b:#010x}"))
                        .collect();
                    assert!(
                        diff.is_empty(),
                        "d={d} bf16={bf16_out} Tkv={over} not clamped to capacity {}: {}",
                        c.cap,
                        diff.join(", ")
                    );
                }
            }
        }
    });
}
