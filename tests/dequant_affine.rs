//! `dequant_affine_{q8,q4}_bf16`: MLX-affine dequantisation to bf16.
//!
//! The reference is the formula `mx.dequantize` computes on the GPU stream:
//! `bf16_rne(fma(f32(scale), f32(q), f32(bias)))`, one rounding to f32 and one
//! to bf16 (the product is exact, so a fused and an unfused multiply-add agree
//! except where the f32 product alone overflows). Two host models are kept
//! apart because the device differs from the IEEE one:
//!
//! * [`Model::Ieee`]: subnormals kept.
//! * [`Model::Device`]: the GPU's fast-math flush: a subnormal scale, bias or
//!   f32 result is a zero of its own sign; NaN is the canonical `0x7fc0`.
//!
//! `tests/fixtures/dequant_affine/` holds `mx.dequantize(stream=mx.gpu)`
//! outputs (`scripts/gen_dequant_affine_fixtures.py`, mlx 0.31.2) at group
//! sizes 32, 64 and 128, for q8 and q4, from `mx.quantize` and from crafted
//! corner values. The kernel agrees with them bit for bit, including on
//! subnormal and overflowing corners, and the device model reproduces every
//! crafted one of them while the IEEE model does not: the flush is mlx's
//! behaviour too, not a quirk of this kernel. Each golden test also checks the
//! fixtures' provenance (`assert_fixture_provenance`): the manifest's
//! `generator_commit` must be on this history and be the commit that last
//! changed the generator, and any check that cannot run fails.
//!
//! The fault-word tests cover `check_dequant_affine_fault`, the checked read of
//! the word `dequant_affine_bf16` raises.
//!
//! All tests need the device (`with_gpu`), so none belongs in the hosted
//! device-free selection.

mod common;

#[path = "../src/bin/common/json.rs"]
mod json;

use common::{with_gpu, with_two_gpus};
use json::{Json, Syntax};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tessl::dequant::{
    check_dequant_affine_fault, dequant_affine_bf16, dequant_affine_q4_bf16, dequant_affine_q8_bf16,
    validate_dequant_affine_bf16, AffineBits, DequantAffineBank, DequantAffineFault, DEQUANT_AFFINE_GROUP_SIZES,
};
use tessl::nn::QuantShape;
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits, f32_to_bf16_bits_gpu, GpuBuffer};
use tessl::GpuRuntime;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Model {
    Ieee,
    Device,
}

fn flush(x: f32) -> f32 {
    if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
        0.0f32.copysign(x)
    } else {
        x
    }
}

/// One element under `model`, from the bf16 bits of scale and bias and the code.
fn model_elem(model: Model, scale: u16, q: u32, bias: u16) -> u16 {
    let (s, b) = (bf16_bits_to_f32(scale), bf16_bits_to_f32(bias));
    match model {
        Model::Ieee => f32_to_bf16_bits(s.mul_add(q as f32, b)),
        Model::Device => f32_to_bf16_bits_gpu(flush(flush(s).mul_add(q as f32, flush(b)))),
    }
}

fn code(packed: &[u32], bits: u32, row: usize, col: usize, cols: usize) -> u32 {
    let per = (32 / bits) as usize;
    let word = packed[row * (cols / per) + col / per];
    (word >> (bits * (col % per) as u32)) & ((1u32 << bits) - 1)
}

#[allow(clippy::too_many_arguments)]
fn model_matrix(
    model: Model,
    bits: u32,
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    rows: usize,
    cols: usize,
    gs: usize,
) -> Vec<u16> {
    let groups = cols / gs;
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let g = r * groups + c / gs;
            out.push(model_elem(model, scales[g], code(packed, bits, r, c, cols), biases[g]));
        }
    }
    out
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A random finite bf16 with a biased exponent in `100..=140` (about 1e-8 to
    /// 1e4): ordinary weights' scales and biases, nowhere near a subnormal.
    fn ordinary(&mut self) -> u16 {
        let r = self.next();
        let sign = (r & 1) as u16;
        let exp = 100 + ((r >> 1) % 41) as u16;
        let mant = ((r >> 8) & 0x7f) as u16;
        (sign << 15) | (exp << 7) | mant
    }
}

struct Case {
    bits: u32,
    rows: usize,
    cols: usize,
    gs: usize,
    packed: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
}

impl Case {
    fn shape(&self) -> QuantShape {
        QuantShape {
            rows: self.rows as u32,
            cols: self.cols as u32,
            group_size: self.gs as u32,
        }
    }

    fn words(bits: u32, rows: usize, cols: usize) -> usize {
        rows * cols * bits as usize / 32
    }

    fn random(rng: &mut Rng, bits: u32, rows: usize, cols: usize, gs: usize) -> Case {
        let packed = (0..Case::words(bits, rows, cols)).map(|_| rng.next() as u32).collect();
        let n = rows * (cols / gs);
        Case {
            bits,
            rows,
            cols,
            gs,
            packed,
            scales: (0..n).map(|_| rng.ordinary()).collect(),
            biases: (0..n).map(|_| rng.ordinary()).collect(),
        }
    }

    fn model(&self, model: Model) -> Vec<u16> {
        model_matrix(
            model,
            self.bits,
            &self.packed,
            &self.scales,
            &self.biases,
            self.rows,
            self.cols,
            self.gs,
        )
    }
}

fn bf16_buffer(rt: &GpuRuntime, bits: &[u16]) -> GpuBuffer {
    let b = rt.alloc_buffer(bits.len().max(1) * 2).expect("alloc bf16");
    b.write_bf16_bits(bits);
    b
}

fn u32_buffer(rt: &GpuRuntime, words: &[u32]) -> GpuBuffer {
    rt.alloc_buffer_from_u32(words).expect("alloc u32")
}

struct Buffers {
    packed: GpuBuffer,
    scales: GpuBuffer,
    biases: GpuBuffer,
    /// A cleared fault word.
    fault: GpuBuffer,
}

impl Buffers {
    fn upload(rt: &GpuRuntime, c: &Case) -> Buffers {
        Buffers {
            packed: u32_buffer(rt, &c.packed),
            scales: bf16_buffer(rt, &c.scales),
            biases: bf16_buffer(rt, &c.biases),
            fault: {
                let f = rt.alloc_buffer(4).expect("alloc fault");
                f.write_u32(&[0]);
                f
            },
        }
    }

    fn bank(&self) -> DequantAffineBank<'_> {
        DequantAffineBank {
            packed: &self.packed,
            scales: &self.scales,
            biases: &self.biases,
        }
    }
}

const SENTINEL: u16 = 0xDEAD;

fn ab(bits: u32) -> AffineBits {
    match bits {
        8 => AffineBits::Q8,
        4 => AffineBits::Q4,
        other => panic!("no {other}-bit affine layout"),
    }
}

fn non_finite(bf16: u16) -> bool {
    bf16 & 0x7F80 == 0x7F80
}

/// Run the kernel for `c` into a fresh scratch of `rows * cols + spare`
/// elements pre-filled with [`SENTINEL`] and a fault word holding `fault_in`;
/// check that exactly one dispatch was encoded and the spare tail was left
/// alone, and return the matrix and the fault word.
fn run_with_fault(rt: &Arc<GpuRuntime>, c: &Case, spare: usize, fault_in: u32) -> (Vec<u16>, u32) {
    let bufs = Buffers::upload(rt, c);
    let n = c.rows * c.cols;
    let out = rt.alloc_buffer((n + spare).max(1) * 2).expect("alloc out");
    out.write_bf16_bits(&vec![SENTINEL; n + spare]);
    let fault = rt.alloc_buffer(4).expect("alloc fault");
    fault.write_u32(&[fault_in]);
    rt.take_dispatch_count();
    let call = if c.bits == 8 {
        dequant_affine_q8_bf16
    } else {
        dequant_affine_q4_bf16
    };
    call(rt, bufs.bank(), &out, &fault, c.shape()).expect("dequant");
    assert_eq!(rt.take_dispatch_count(), 1, "exactly one dispatch");
    rt.synchronize().expect("sync");
    let got = out.contents_u16()[..n + spare].to_vec();
    assert!(
        got[n..].iter().all(|&v| v == SENTINEL),
        "the kernel wrote past rows * cols into the caller's scratch"
    );
    (got[..n].to_vec(), fault.read_u32()[0])
}

/// [`run_with_fault`] from a clear fault word, asserting the fault contract on
/// every run: the word is 1 exactly when some output element is non-finite.
fn run(rt: &Arc<GpuRuntime>, c: &Case, spare: usize) -> Vec<u16> {
    let (got, fault) = run_with_fault(rt, c, spare, 0);
    let any = got.iter().any(|&v| non_finite(v));
    assert_eq!(
        fault,
        u32::from(any),
        "fault word {fault} but {} non-finite outputs",
        got.iter().filter(|&&v| non_finite(v)).count()
    );
    got
}

fn first_diff(got: &[u16], want: &[u16]) -> Option<usize> {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).position(|(g, w)| g != w)
}

fn assert_bits(what: &str, got: &[u16], want: &[u16]) {
    if let Some(i) = first_diff(got, want) {
        let n = got.iter().zip(want).filter(|(g, w)| g != w).count();
        panic!(
            "{what}: {n} of {} elements differ; first at {i}: kernel {:#06x}, model {:#06x}",
            got.len(),
            got[i],
            want[i]
        );
    }
}

// ---------------------------------------------------------------- parity ---

/// Every distinct 2-D language-model weight shape in Qwen3.5-2B-Base's
/// safetensors header (read from the header, not derived from the config), but
/// the 248320 x 2048 tied embedding (a 1 GiB output, which the 6144 x 2048 case
/// covers row for row): in_proj_a/b 16x2048, k/v_proj 512x2048, o_proj and
/// linear in_proj_z/out_proj 2048x2048, q_proj 4096x2048, in_proj_qkv and MLP
/// gate/up 6144x2048, down_proj 2048x6144. Plus the vision tower's 1024x4096
/// and 4096x1024, which put other widths through the same kernel.
const QWEN35_2B_SHAPES: [(usize, usize); 9] = [
    (6144, 2048),
    (2048, 6144),
    (2048, 2048),
    (4096, 2048),
    (512, 2048),
    (16, 2048),
    (1024, 4096),
    (4096, 1024),
    (1, 2048),
];

#[test]
fn matches_the_host_formula_bit_for_bit_on_the_2b_shapes() {
    with_gpu(|rt| {
        let mut rng = Rng(0x2b);
        for bits in [8u32, 4] {
            for (rows, cols) in QWEN35_2B_SHAPES {
                let c = Case::random(&mut rng, bits, rows, cols, 64);
                let got = run(rt, &c, 0);
                // Ordinary scales and biases never touch a subnormal, so the two
                // host models coincide here and the kernel must equal both.
                let ieee = c.model(Model::Ieee);
                assert_bits(&format!("q{bits} {rows}x{cols} vs IEEE"), &got, &ieee);
                assert_eq!(
                    ieee,
                    c.model(Model::Device),
                    "q{bits} {rows}x{cols}: models differ on ordinary data"
                );
            }
        }
    });
}

#[test]
fn matches_on_odd_row_counts_every_group_size_and_a_reused_oversized_scratch() {
    with_gpu(|rt| {
        let mut rng = Rng(0x0dd);
        assert_eq!(DEQUANT_AFFINE_GROUP_SIZES, [32, 64, 128]);
        // One scratch for every shape, as ojas-metal reuses one across layers;
        // sized for the largest and refilled with the sentinel before each use.
        let scratch = rt.alloc_buffer(65 * 256 * 2).expect("scratch");
        let fault = rt.alloc_buffer(4).expect("fault");
        for bits in [8u32, 4] {
            for gs in DEQUANT_AFFINE_GROUP_SIZES {
                for (rows, cols) in [(1usize, 256usize), (3, 128), (65, 256), (2, 32 * 4)] {
                    if !cols.is_multiple_of(gs as usize) {
                        continue;
                    }
                    let c = Case::random(&mut rng, bits, rows, cols, gs as usize);
                    let bufs = Buffers::upload(rt, &c);
                    let n = rows * cols;
                    scratch.write_bf16_bits(&vec![SENTINEL; 65 * 256]);
                    fault.write_u32(&[0]);
                    dequant_affine_bf16(rt, ab(bits), bufs.bank(), &scratch, &fault, c.shape()).expect("dequant");
                    rt.synchronize().expect("sync");
                    assert_eq!(fault.read_u32()[0], 0, "ordinary data must not raise the fault word");
                    let all = scratch.contents_u16()[..65 * 256].to_vec();
                    assert_bits(
                        &format!("q{bits} g{gs} {rows}x{cols}"),
                        &all[..n],
                        &c.model(Model::Ieee),
                    );
                    assert!(
                        all[n..].iter().all(|&v| v == SENTINEL),
                        "q{bits} g{gs} {rows}x{cols}: wrote past the matrix in a larger scratch"
                    );
                }
            }
        }
    });
}

#[test]
fn zero_rows_is_a_validated_no_op() {
    with_gpu(|rt| {
        let (packed, scales, biases, out) = (
            rt.alloc_buffer(8).expect("packed"),
            rt.alloc_buffer(8).expect("scales"),
            rt.alloc_buffer(8).expect("biases"),
            rt.alloc_buffer(8).expect("out"),
        );
        let fault = rt.alloc_buffer(4).expect("fault");
        fault.write_u32(&[0]);
        let bank = DequantAffineBank {
            packed: &packed,
            scales: &scales,
            biases: &biases,
        };
        let shape = QuantShape {
            rows: 0,
            cols: 64,
            group_size: 64,
        };
        rt.take_dispatch_count();
        dequant_affine_q8_bf16(rt, bank, &out, &fault, shape).expect("zero rows");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

// -------------------------------------------------------------- rounding ---

/// Independent of the host conversion helper: `256 + q` (q in 0..=255) in bf16,
/// whose spacing is 2 there, so every odd sum is an exact tie. Round half to
/// even on the mantissa index `m = (v - 256) / 2` in integers.
fn expected_256_plus(q: u32) -> u16 {
    let k = q; // v - 256
    let mut m = k / 2;
    if k % 2 == 1 && m % 2 == 1 {
        m += 1;
    }
    0x4380 + m as u16 // 256.0 is 0x4380; a carry into the exponent is the same add
}

#[test]
fn float_to_bf16_conversion_is_round_to_nearest_even_on_exact_ties() {
    with_gpu(|rt| {
        for bits in [8u32, 4] {
            let per = 32 / bits;
            let qmax = (1u32 << bits) - 1;
            // One row of one group of 64: codes 0..=qmax cycling, scale 1.0, bias 256.0.
            let codes: Vec<u32> = (0..64).map(|i| (i as u32) % (qmax + 1)).collect();
            let mut words = vec![0u32; 64 / per as usize];
            for (i, q) in codes.iter().enumerate() {
                words[i / per as usize] |= q << (bits * (i as u32 % per));
            }
            for (sign, name) in [(0u16, "positive"), (0x8000, "negative")] {
                let c = Case {
                    bits,
                    rows: 1,
                    cols: 64,
                    gs: 64,
                    packed: words.clone(),
                    scales: vec![0x3F80 | sign], // +-1.0
                    biases: vec![0x4380 | sign], // +-256.0
                };
                let got = run(rt, &c, 0);
                for (i, q) in codes.iter().enumerate() {
                    assert_eq!(
                        got[i],
                        expected_256_plus(*q) | sign,
                        "q{bits} {name}: 256 + {q} should round half to even"
                    );
                }
            }
        }
    });
}

#[test]
fn a_tie_at_bf16_max_rounds_up_to_infinity_and_a_value_below_it_does_not() {
    with_gpu(|rt| {
        // max_finite + 2^119 is exactly half a bf16 ulp above the largest finite
        // bf16 (mantissa 0x7f, odd), so RNE carries to infinity. One code step
        // lower stays finite.
        let c = Case {
            bits: 8,
            rows: 1,
            cols: 64,
            gs: 64,
            packed: (0..16).map(|_| 0x0101_0101u32).collect(), // every code 1
            scales: vec![0x7F7F],                              // bf16 max
            biases: vec![0x7B00],                              // 2^119, the half ulp
        };
        let got = run(rt, &c, 0);
        assert!(
            got.iter().all(|&v| v == 0x7F80),
            "tie at max must round to +inf: {:#06x}",
            got[0]
        );
        let below = Case {
            biases: vec![0x7A80], // 2^118: a quarter ulp, rounds back down
            ..c
        };
        let got = run(rt, &below, 0);
        assert!(
            got.iter().all(|&v| v == 0x7F7F),
            "below the tie stays max: {:#06x}",
            got[0]
        );
    });
}

// ------------------------------------------------------ non-finite values ---

#[test]
fn finite_inputs_that_overflow_bf16_write_infinity_and_raise_the_fault_word() {
    with_gpu(|rt| {
        // scale = bf16 max, bias 0: code 0 -> 0, code 1 -> max, code >= 2 -> the
        // f32 product itself overflows. Negative scale mirrors it.
        let codes = [0u32, 1, 2, 3, 255, 128, 7, 100];
        let mut words = vec![0u32; 8];
        for i in 0..32 {
            words[i / 4] |= codes[i % 8] << (8 * (i % 4));
        }
        for (scale, one, inf) in [(0x7F7Fu16, 0x7F7Fu16, 0x7F80u16), (0xFF7F, 0xFF7F, 0xFF80)] {
            let c = Case {
                bits: 8,
                rows: 1,
                cols: 32,
                gs: 32,
                packed: words.clone(),
                scales: vec![scale],
                biases: vec![0],
            };
            let got = run(rt, &c, 0);
            assert_bits("overflow row", &got, &c.model(Model::Device));
            assert_eq!(got[0], 0, "code 0 -> zero");
            assert_eq!(got[1], one, "code 1 -> +-max");
            assert!(
                got[2..8].iter().all(|&v| v == inf),
                "codes >= 2 -> +-inf: {:?}",
                &got[..8]
            );
        }
        // Beyond the f32 range as well: 3 * 2^126 + 2^127.
        let c = Case {
            bits: 8,
            rows: 1,
            cols: 32,
            gs: 32,
            packed: vec![0x0303_0303; 8],
            scales: vec![0x7E80], // 2^126
            biases: vec![0x7F00], // 2^127
        };
        let got = run(rt, &c, 0);
        assert!(
            got.iter().all(|&v| v == 0x7F80),
            "3 * 2^126 + 2^127 is beyond bf16: {:#06x}",
            got[0]
        );
    });
}

#[test]
fn an_exact_fma_keeps_a_product_that_overflows_f32_when_the_bias_cancels_it() {
    with_gpu(|rt| {
        // 2^125 * 8 = 2^128 overflows an f32 product, but fma(2^125, 8, -max) is
        // finite: the kernel fuses, as mx.dequantize does. The crafted q4 mlx golden
        // below also holds this corner (group 168: fused and unfused differ on 34
        // elements); the q8 one does not.
        let c = Case {
            bits: 8,
            rows: 1,
            cols: 32,
            gs: 32,
            packed: vec![0x0808_0808; 8],
            scales: vec![0x7E00], // 2^125
            biases: vec![0xFF7F], // -bf16 max
        };
        let got = run(rt, &c, 0);
        let want = c.model(Model::Ieee);
        assert_bits("fma overflow-cancel", &got, &want);
        assert!(
            got[0] != 0x7F80 && got[0] & 0x7F80 != 0x7F80,
            "must be finite: {:#06x}",
            got[0]
        );
    });
}

#[test]
fn non_finite_scales_and_biases_propagate_by_ieee_rules() {
    with_gpu(|rt| {
        // Codes 0..=7 repeated; scale/bias each from {+inf, -inf, NaN, 1.0}.
        let specials = [0x7F80u16, 0xFF80, 0x7FC0, 0x3F80];
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        for s in specials {
            for b in specials {
                scales.push(s);
                biases.push(b);
            }
        }
        // 16 groups of 32 columns, one row.
        let words: Vec<u32> = (0..16 * 8).map(|_| 0x0302_0100u32).collect();
        let c = Case {
            bits: 8,
            rows: 1,
            cols: 16 * 32,
            gs: 32,
            packed: words,
            scales,
            biases,
        };
        let got = run(rt, &c, 0);
        assert_bits("non-finite inputs", &got, &c.model(Model::Device));
        // inf * 0 and inf - inf are NaN, never a finite value in disguise.
        assert!(got.contains(&0x7FC0), "no NaN produced from inf*0 / inf-inf");
        assert!(got.contains(&0x7F80) && got.contains(&0xFF80));
    });
}

// ------------------------------------------------------------ subnormals ---

#[test]
fn subnormals_flush_to_signed_zero_where_the_ieee_formula_does_not() {
    with_gpu(|rt| {
        // Hand-picked: (scale, bias, code) -> what IEEE says, what the device does.
        // scale 0x0001 = 2^-133 (smallest bf16 subnormal): 3 * 2^-133 is exactly
        // representable, IEEE keeps it as bf16 0x0003; the device flushes the
        // scale, so 0 * 3 + 0.
        let picks: [(u16, u16, u32, u16, u16); 5] = [
            (0x0001, 0x0000, 3, 0x0003, 0x0000),
            (0x8001, 0x0000, 3, 0x8003, 0x0000),
            // Normal scale, subnormal bias: bias flushed, result is exactly scale * q.
            (0x3F80, 0x0040, 2, 0x4000, 0x4000),
            // Normal operands, subnormal result: 2^-126 * 1 - (2^-126)(1 + 2^-7) = -2^-133.
            (0x0080, 0x8081, 1, 0x8001, 0x8000),
            // Subnormal bias alone with a zero scale.
            (0x0000, 0x8001, 5, 0x8001, 0x0000),
        ];
        for (scale, bias, q, ieee, device) in picks {
            assert_eq!(
                model_elem(Model::Ieee, scale, q, bias),
                ieee,
                "IEEE model, scale {scale:#06x} bias {bias:#06x}"
            );
            assert_eq!(
                model_elem(Model::Device, scale, q, bias),
                device,
                "device model, scale {scale:#06x}"
            );
            for bits in [8u32, 4] {
                let c = Case {
                    bits,
                    rows: 1,
                    cols: 64,
                    gs: 64,
                    packed: {
                        let per = 32 / bits;
                        let mut w = vec![0u32; 64 / per as usize];
                        for i in 0..64u32 {
                            w[(i / per) as usize] |= q << (bits * (i % per));
                        }
                        w
                    },
                    scales: vec![scale],
                    biases: vec![bias],
                };
                let got = run(rt, &c, 0);
                assert!(
                    got.iter().all(|&v| v == device),
                    "q{bits}: scale {scale:#06x} bias {bias:#06x} q {q}: device gave {:#06x}, want {device:#06x} (IEEE {ieee:#06x})",
                    got[0]
                );
            }
        }
    });
}

// ----------------------------------------------------------- mlx goldens ---

const GROUP: usize = 64;
const ROWS: usize = 17;
const COLS: usize = 1024;
const GENERATOR: &str = "scripts/gen_dequant_affine_fixtures.py";
/// Longest any one git call may take before provenance is "could not verify".
const GIT_DEADLINE: Duration = Duration::from_secs(30);

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/dequant_affine")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn u16s(name: &str) -> Vec<u16> {
    let raw = fixture(name);
    let (halves, rest) = raw.as_chunks::<2>();
    assert!(rest.is_empty(), "{name}: not a whole number of u16");
    halves.iter().map(|b| u16::from_le_bytes(*b)).collect()
}

fn u32s(name: &str) -> Vec<u32> {
    let raw = fixture(name);
    let (words, rest) = raw.as_chunks::<4>();
    assert!(rest.is_empty(), "{name}: not a whole number of u32");
    words.iter().map(|b| u32::from_le_bytes(*b)).collect()
}

/// One golden: its inputs as a [`Case`] and mlx's output bits. Group size 64 keeps
/// the original file names; the others carry `_g{gs}`.
fn golden_case(case: &str, bits: u32, gs: usize) -> (Case, Vec<u16>) {
    let name = |part: &str| {
        if gs == GROUP {
            format!("{case}_q{bits}_{part}.bin")
        } else {
            format!("{case}_q{bits}_g{gs}_{part}.bin")
        }
    };
    let c = Case {
        bits,
        rows: ROWS,
        cols: COLS,
        gs,
        packed: u32s(&name("packed")),
        scales: u16s(&name("scales")),
        biases: u16s(&name("biases")),
    };
    assert_eq!(
        c.packed.len(),
        Case::words(bits, ROWS, COLS),
        "{case} q{bits} g{gs} packed length"
    );
    assert_eq!(c.scales.len(), ROWS * COLS / gs);
    assert_eq!(c.biases.len(), ROWS * COLS / gs);
    let want = u16s(&name("out"));
    assert_eq!(want.len(), ROWS * COLS);
    (c, want)
}

/// Run git in the repo, with a deadline: `(exit code, stdout)`. A git that is
/// missing, killed by a signal or past the deadline is an `Err`.
fn git(args: &[&str]) -> Result<(i32, String), String> {
    let mut child = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("git {args:?} did not start: {e}"))?;
    let deadline = Instant::now() + GIT_DEADLINE;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("git {args:?} still running after {GIT_DEADLINE:?}; killed"));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("git {args:?} could not be waited on: {e}"));
            }
        }
    };
    let mut out = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        std::io::Read::read_to_string(&mut pipe, &mut out)
            .map_err(|e| format!("git {args:?} output unreadable: {e}"))?;
    }
    let code = status
        .code()
        .ok_or_else(|| format!("git {args:?} was killed by a signal"))?;
    Ok((code, out.trim().to_string()))
}

/// The provenance verdict for a manifest pin `pin`, given `run` (git). `Ok` only
/// when the pin is a commit on this history and is the commit that last changed
/// the generator; every other outcome, including one where a check could not
/// run, is an `Err` naming it.
fn judge_provenance(pin: &str, run: &dyn Fn(&[&str]) -> Result<(i32, String), String>) -> Result<(), String> {
    let unverified = |why: String| format!("could not verify provenance: {why}");
    let (code, shallow) = run(&["rev-parse", "--is-shallow-repository"]).map_err(unverified)?;
    if code != 0 || shallow != "false" {
        return Err(unverified(format!(
            "this is not a full clone (rev-parse --is-shallow-repository exited {code}, printed {shallow:?})"
        )));
    }
    let commit = format!("{pin}^{{commit}}");
    let (code, _) = run(&["cat-file", "-e", &commit]).map_err(unverified)?;
    if code != 0 {
        return Err(unverified(format!(
            "commit {pin} is not in this repository (cat-file -e exited {code})"
        )));
    }
    let (code, _) = run(&["merge-base", "--is-ancestor", pin, "HEAD"]).map_err(unverified)?;
    match code {
        0 => {}
        1 => return Err(format!("generator_commit {pin} is not on this history")),
        other => return Err(unverified(format!("merge-base --is-ancestor exited {other}"))),
    }
    let (code, last) = run(&["log", "-1", "--format=%H", "--", GENERATOR]).map_err(unverified)?;
    if code != 0 || last.len() != 40 {
        return Err(unverified(format!(
            "git log of {GENERATOR} exited {code} and printed {last:?}"
        )));
    }
    if last != pin {
        return Err(format!(
            "generator changed after the fixtures: the manifest pins {pin}, the generator was last changed in {last}; regenerate"
        ));
    }
    Ok(())
}

/// The manifest's `generator_commit` of the fixture directory.
fn manifest_pin() -> Result<String, String> {
    let text = String::from_utf8(fixture("MANIFEST.json")).map_err(|e| format!("MANIFEST.json is not UTF-8: {e}"))?;
    let doc = json::parse(
        &text,
        Syntax {
            what: "MANIFEST.json",
            max_depth: 8,
            uints_only: false,
            literals: true,
        },
    )
    .map_err(|e| format!("MANIFEST.json does not parse: {e}"))?;
    match doc.get("generator_commit") {
        Some(Json::Str(pin))
            if pin.len() == 40 && pin.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
        {
            Ok(pin.clone())
        }
        other => Err(format!(
            "MANIFEST.json generator_commit is not a 40-hex commit: {other:?}"
        )),
    }
}

/// Panics unless the goldens were written by the generator as it is now, from a
/// commit on this history. Called inside `with_gpu`, so it never runs in the
/// hosted device-free selection (a hosted clone is shallow and could not verify).
fn assert_fixture_provenance() {
    let verdict = manifest_pin().and_then(|pin| judge_provenance(&pin, &git));
    if let Err(why) = verdict {
        panic!("dequant_affine fixtures: {why}");
    }
}

/// The kernel against one mlx golden, plus what the golden must contain.
fn assert_matches_mlx(rt: &Arc<GpuRuntime>, case: &str, bits: u32, gs: usize, c: &Case, want: &[u16]) {
    let what = format!("{case} q{bits} g{gs}");
    assert_bits(&format!("{what} vs mlx"), &run(rt, c, 0), want);
    if case == "crafted" {
        // The goldens must contain the corners they are there for.
        assert!(
            want.iter().any(|&v| v & 0x7FFF == 0x7F80),
            "no infinity in the {what} golden"
        );
        assert!(
            want.contains(&0x0000) && want.contains(&0x8000),
            "no zeros of both signs in the {what} golden"
        );
        // The device model reproduces mlx exactly; the IEEE one does not
        // (subnormal operands and results), which is why the kernel is not
        // compared against it here.
        assert_eq!(c.model(Model::Device), want, "{what}: device model vs mlx");
        let ieee_misses = c.model(Model::Ieee).iter().zip(want).filter(|(a, b)| a != b).count();
        assert!(ieee_misses > 0, "{what}: the golden has no subnormal case");
    }
}

#[test]
fn matches_mlx_gpu_dequantize_bit_for_bit() {
    // Real mx.quantize output (weights spanning ~1e-30 to 1e3, an all-negative
    // row, a constant row and a zero row), then random packed words with scales
    // and biases drawn from the corners: signed zeros, subnormals, the smallest
    // normal, +-max, 2^125.
    let goldens: Vec<_> = [8u32, 4]
        .into_iter()
        .flat_map(|bits| ["quant", "crafted"].map(|case| (case, bits, golden_case(case, bits, GROUP))))
        .collect();
    with_gpu(|rt| {
        assert_fixture_provenance();
        for (case, bits, (c, want)) in &goldens {
            assert_matches_mlx(rt, case, *bits, GROUP, c, want);
        }
    });
}

#[test]
fn matches_mlx_gpu_dequantize_at_group_sizes_32_and_128() {
    let goldens: Vec<_> = [32usize, 128]
        .into_iter()
        .flat_map(|gs| {
            [8u32, 4].into_iter().flat_map(move |bits| {
                ["quant", "crafted"].map(move |case| (case, bits, gs, golden_case(case, bits, gs)))
            })
        })
        .collect();
    with_gpu(|rt| {
        assert_fixture_provenance();
        for (case, bits, gs, (c, want)) in &goldens {
            assert_matches_mlx(rt, case, *bits, *gs, c, want);
        }
    });
}

#[test]
fn provenance_passes_only_for_an_ancestor_that_is_the_generators_last_touch() {
    let pin = "a".repeat(40);
    let other = "b".repeat(40);
    // A scripted git: `shallow`, `has_commit`, `ancestor`, `last` pick each answer.
    let scripted = |shallow: &'static str, has_commit: i32, ancestor: i32, last: String| {
        move |args: &[&str]| -> Result<(i32, String), String> {
            Ok(match args[0] {
                "rev-parse" => (0, shallow.to_string()),
                "cat-file" => (has_commit, String::new()),
                "merge-base" => (ancestor, String::new()),
                "log" => (0, last.clone()),
                other => panic!("unexpected git {other}"),
            })
        }
    };
    judge_provenance(&pin, &scripted("false", 0, 0, pin.clone())).expect("the one passing outcome");

    let fails = |run: &dyn Fn(&[&str]) -> Result<(i32, String), String>, needle: &str| {
        let e = judge_provenance(&pin, run).expect_err("must not pass");
        assert!(e.contains(needle), "wanted {needle:?} in {e:?}");
    };
    fails(&scripted("false", 0, 1, pin.clone()), "not on this history");
    fails(
        &scripted("false", 0, 0, other.clone()),
        "generator changed after the fixtures",
    );
    fails(&scripted("true", 0, 0, pin.clone()), "could not verify provenance");
    fails(&scripted("", 0, 0, pin.clone()), "could not verify provenance");
    fails(&scripted("false", 128, 0, pin.clone()), "could not verify provenance");
    fails(&scripted("false", 0, 128, pin.clone()), "could not verify provenance");
    fails(&scripted("false", 0, 0, String::new()), "could not verify provenance");
    fails(&|_| Err("git is missing".into()), "could not verify provenance");
    fails(
        &|args| {
            if args[0] == "rev-parse" {
                Ok((0, "false".into()))
            } else {
                Err("deadline".into())
            }
        },
        "could not verify provenance",
    );
    // The real check on this checkout, against the real git. Inside `with_gpu` so that this
    // test, like every test here, is outside the hosted device-free selection.
    with_gpu(|_| assert_fixture_provenance());
}

// -------------------------------------------------------------- refusals ---

fn tiny(rt: &GpuRuntime, rows: usize, cols: usize, gs: usize, bits: u32) -> (Case, Buffers, GpuBuffer) {
    let c = Case::random(&mut Rng(9), bits, rows, cols, gs);
    let b = Buffers::upload(rt, &c);
    let out = rt.alloc_buffer((rows * cols).max(1) * 2).expect("out");
    (c, b, out)
}

fn refuse(
    rt: &Arc<GpuRuntime>,
    bits: AffineBits,
    bank: DequantAffineBank<'_>,
    out: &GpuBuffer,
    fault: &GpuBuffer,
    s: QuantShape,
) -> String {
    rt.take_dispatch_count();
    let err = dequant_affine_bf16(rt, bits, bank, out, fault, s).expect_err("must refuse");
    assert_eq!(rt.take_dispatch_count(), 0, "a refusal encoded work: {err}");
    let v = validate_dequant_affine_bf16(rt, bits, bank, out, fault, s).expect_err("validate must refuse too");
    assert_eq!(err, v, "validate and dispatch disagree");
    err
}

#[test]
fn unsupported_group_sizes_and_ragged_columns_are_refused() {
    with_gpu(|rt| {
        let (c, b, out) = tiny(rt, 2, 128, 64, 8);
        for gs in [0u32, 1, 8, 16, 48, 96, 256, 512] {
            let e = refuse(
                rt,
                AffineBits::Q8,
                b.bank(),
                &out,
                &b.fault,
                QuantShape {
                    group_size: gs,
                    ..c.shape()
                },
            );
            assert!(e.contains("group_size"), "{e}");
        }
        let e = refuse(
            rt,
            AffineBits::Q8,
            b.bank(),
            &out,
            &b.fault,
            QuantShape { cols: 96, ..c.shape() },
        );
        assert!(e.contains("not a multiple of group_size"), "{e}");
    });
}

#[test]
fn a_buffer_one_element_short_is_refused_for_each_operand_and_both_widths() {
    with_gpu(|rt| {
        for bits in [8u32, 4] {
            let ab = ab(bits);
            let (c, b, out) = tiny(rt, 3, 128, 64, bits);
            // Sanity: the exact-fit buffers are accepted.
            validate_dequant_affine_bf16(rt, ab, b.bank(), &out, &b.fault, c.shape()).expect("exact fit");
            let short_packed = rt.alloc_buffer(c.packed.len() * 4 - 4).expect("alloc");
            let short_scales = rt.alloc_buffer(c.scales.len() * 2 - 2).expect("alloc");
            let short_biases = rt.alloc_buffer(c.biases.len() * 2 - 2).expect("alloc");
            let short_out = rt.alloc_buffer(3 * 128 * 2 - 2).expect("alloc");
            let cases: [(&str, DequantAffineBank<'_>, &GpuBuffer); 4] = [
                (
                    "packed",
                    DequantAffineBank {
                        packed: &short_packed,
                        ..b.bank()
                    },
                    &out,
                ),
                (
                    "scales",
                    DequantAffineBank {
                        scales: &short_scales,
                        ..b.bank()
                    },
                    &out,
                ),
                (
                    "biases",
                    DequantAffineBank {
                        biases: &short_biases,
                        ..b.bank()
                    },
                    &out,
                ),
                ("out", b.bank(), &short_out),
            ];
            for (which, bank, o) in cases {
                let e = refuse(rt, ab, bank, o, &b.fault, c.shape());
                assert!(e.contains(which), "q{bits}: error should name {which}: {e}");
            }
        }
    });
}

#[test]
fn a_shape_too_large_for_metal_indexing_is_refused_before_any_buffer_is_read() {
    with_gpu(|rt| {
        let (_, b, out) = tiny(rt, 1, 64, 64, 4);
        // u32::MAX rows of 64 columns: 2^34 q4 words, far past `uint` indexing.
        let huge = QuantShape {
            rows: u32::MAX,
            cols: 64,
            group_size: 64,
        };
        let e = refuse(rt, AffineBits::Q4, b.bank(), &out, &b.fault, huge);
        assert!(e.contains("uint indexing"), "{e}");
    });
}

#[test]
fn an_output_that_overlaps_an_input_is_refused() {
    with_gpu(|rt| {
        let (c, b, out) = tiny(rt, 2, 128, 64, 8);
        // Each input in turn is also the output; it is allocated big enough for
        // `out` so the overlap check, not the length check, is what refuses.
        let big = rt.alloc_buffer(2 * 128 * 2).expect("big");
        let cases = [
            (
                "packed",
                DequantAffineBank {
                    packed: &big,
                    ..b.bank()
                },
            ),
            (
                "scales",
                DequantAffineBank {
                    scales: &big,
                    ..b.bank()
                },
            ),
            (
                "biases",
                DequantAffineBank {
                    biases: &big,
                    ..b.bank()
                },
            ),
        ];
        for (name, bank) in cases {
            let e = refuse(rt, AffineBits::Q8, bank, &big, &b.fault, c.shape());
            assert!(e.contains("overlap") && e.contains(name), "{e}");
        }
        validate_dequant_affine_bf16(rt, AffineBits::Q8, b.bank(), &out, &b.fault, c.shape())
            .expect("disjoint is fine");
    });
}

#[test]
fn a_buffer_from_another_runtime_is_refused() {
    with_two_gpus(|a, other| {
        let (c, b, out) = tiny(a, 2, 128, 64, 8);
        let foreign_out = other.alloc_buffer(2 * 128 * 2).expect("foreign");
        let e = refuse(a, AffineBits::Q8, b.bank(), &foreign_out, &b.fault, c.shape());
        assert!(e.contains("another runtime"), "{e}");
        // And the same call with its own buffers works.
        dequant_affine_q8_bf16(a, b.bank(), &out, &b.fault, c.shape()).expect("own buffers");
    });
}

#[test]
fn a_fault_word_that_is_too_small_foreign_or_overlapping_is_refused() {
    with_two_gpus(|rt, other| {
        let (c, b, out) = tiny(rt, 2, 128, 64, 8);
        let short = rt.alloc_buffer(2).expect("2 bytes");
        let e = refuse(rt, AffineBits::Q8, b.bank(), &out, &short, c.shape());
        assert!(e.contains("fault"), "{e}");
        let foreign = other.alloc_buffer(4).expect("foreign fault");
        let e = refuse(rt, AffineBits::Q8, b.bank(), &out, &foreign, c.shape());
        assert!(e.contains("another runtime") && e.contains("fault"), "{e}");
        // The fault word is written by the kernel, so it may not be the output
        // (or an input) either.
        let e = refuse(rt, AffineBits::Q8, b.bank(), &out, &out, c.shape());
        assert!(e.contains("overlap"), "{e}");
        let wide = rt.alloc_buffer(2 * 128 * 2).expect("input as fault");
        let bank = DequantAffineBank {
            scales: &wide,
            ..b.bank()
        };
        let e = refuse(rt, AffineBits::Q8, bank, &out, &wide, c.shape());
        assert!(e.contains("overlap") && e.contains("fault"), "{e}");
    });
}

// ------------------------------------------------------------ fault word ---

#[test]
fn the_fault_word_is_raised_by_a_non_finite_output_and_is_never_cleared() {
    with_gpu(|rt| {
        let mut rng = Rng(0xfa);
        let clean = Case::random(&mut rng, 8, 3, 64, 64);
        let (_, fault) = run_with_fault(rt, &clean, 0, 0);
        assert_eq!(fault, 0, "a clean dequant leaves a clear word clear");
        let (_, fault) = run_with_fault(rt, &clean, 0, 1);
        assert_eq!(fault, 1, "a clean dequant must not clear an earlier fault");

        // One element out of 3 x 64 overflows: scale of exactly one group is max,
        // and only the codes >= 2 of that group overflow.
        for bits in [8u32, 4] {
            let mut bad = Case::random(&mut rng, bits, 3, 64, 64);
            bad.scales[1] = 0x7F7F;
            bad.biases[1] = 0;
            let (got, fault) = run_with_fault(rt, &bad, 0, 0);
            assert_eq!(fault, 1, "q{bits}: an infinity in row 1 must raise the fault word");
            assert!(got[64..128].iter().any(|&v| v & 0x7F80 == 0x7F80));
            assert!(
                got[..64].iter().chain(&got[128..]).all(|&v| v & 0x7F80 != 0x7F80),
                "q{bits}: the other rows stay finite"
            );
        }
    });
}

// -------------------------------------------------- checked fault-word read ---

/// Dispatch `c` once into `fault` (whatever it holds) and synchronize.
fn dispatch_and_sync(rt: &Arc<GpuRuntime>, c: &Case, fault: &GpuBuffer) {
    let bufs = Buffers::upload(rt, c);
    let out = rt.alloc_buffer((c.rows * c.cols).max(1) * 2).expect("alloc out");
    dequant_affine_bf16(rt, ab(c.bits), bufs.bank(), &out, fault, c.shape()).expect("dequant");
    rt.synchronize().expect("sync");
}

/// A case with one overflowing group (row 1's scale is bf16 max, bias 0, so every
/// code >= 2 gives an infinity).
fn overflowing(rng: &mut Rng, bits: u32) -> Case {
    let mut bad = Case::random(rng, bits, 3, 64, 64);
    bad.scales[1] = 0x7F7F;
    bad.biases[1] = 0;
    bad
}

#[test]
fn check_returns_ok_for_finite_output_and_raised_for_a_non_finite_one() {
    with_gpu(|rt| {
        let mut rng = Rng(0xc4);
        for bits in [8u32, 4] {
            let fault = rt.alloc_buffer(4).expect("fault");
            fault.try_zero().expect("zero");
            dispatch_and_sync(rt, &Case::random(&mut rng, bits, 3, 64, 64), &fault);
            assert_eq!(check_dequant_affine_fault(&fault), Ok(()), "q{bits}: a clean run");

            dispatch_and_sync(rt, &overflowing(&mut rng, bits), &fault);
            let want = Err(DequantAffineFault::Raised { word: 1 });
            assert_eq!(check_dequant_affine_fault(&fault), want, "q{bits}: a faulting run");
            // Sticky: the check does not clear it, and a clean run after does not either.
            assert_eq!(
                check_dequant_affine_fault(&fault),
                want,
                "q{bits}: the read must not clear"
            );
            dispatch_and_sync(rt, &Case::random(&mut rng, bits, 3, 64, 64), &fault);
            assert_eq!(
                check_dequant_affine_fault(&fault),
                want,
                "q{bits}: a clean run must not clear"
            );
            fault.try_zero().expect("reset");
            assert_eq!(
                check_dequant_affine_fault(&fault),
                Ok(()),
                "q{bits}: after the caller's reset"
            );
        }

        // Only word 0 of a larger buffer is read.
        let wide = rt.alloc_buffer(64).expect("64 bytes");
        let mut words = vec![u32::MAX; 16];
        words[0] = 0;
        wide.write_u32(&words);
        assert_eq!(
            check_dequant_affine_fault(&wide),
            Ok(()),
            "words past the first are not the fault"
        );
        words = vec![0; 16];
        words[0] = 3;
        wide.write_u32(&words);
        assert_eq!(
            check_dequant_affine_fault(&wide),
            Err(DequantAffineFault::Raised { word: 3 })
        );
    });
}

#[test]
fn check_reports_a_stale_word_and_never_clears_it() {
    with_gpu(|rt| {
        let mut rng = Rng(0x57a1e);
        let fault = rt.alloc_buffer(4).expect("fault");
        fault.write_u32(&[0xABCD]);
        // A clean dequant leaves the caller's word alone, so a word that was not
        // zeroed first reads as raised: stale and new faults are indistinguishable.
        dispatch_and_sync(rt, &Case::random(&mut rng, 8, 3, 64, 64), &fault);
        assert_eq!(
            fault.read_u32()[0],
            0xABCD,
            "a clean dequant leaves the word as it found it"
        );
        assert_eq!(
            check_dequant_affine_fault(&fault),
            Err(DequantAffineFault::Raised { word: 0xABCD })
        );
        assert_eq!(fault.read_u32()[0], 0xABCD, "the check never writes");
        // A faulting dequant stores 1 over it (a store, not an OR).
        dispatch_and_sync(rt, &overflowing(&mut rng, 8), &fault);
        assert_eq!(fault.read_u32()[0], 1, "a fault overwrites the caller's code with 1");
        assert_eq!(
            check_dequant_affine_fault(&fault),
            Err(DequantAffineFault::Raised { word: 1 })
        );

        // Zero rows dispatches nothing, so the check reads whatever the caller left.
        let (packed, scales, biases, out) = (
            rt.alloc_buffer(8).expect("packed"),
            rt.alloc_buffer(8).expect("scales"),
            rt.alloc_buffer(8).expect("biases"),
            rt.alloc_buffer(8).expect("out"),
        );
        let bank = DequantAffineBank {
            packed: &packed,
            scales: &scales,
            biases: &biases,
        };
        let shape = QuantShape {
            rows: 0,
            cols: 64,
            group_size: 64,
        };
        fault.write_u32(&[0x77]);
        dequant_affine_q8_bf16(rt, bank, &out, &fault, shape).expect("zero rows");
        assert_eq!(
            check_dequant_affine_fault(&fault),
            Err(DequantAffineFault::Raised { word: 0x77 })
        );
    });
}

#[test]
fn check_on_a_poisoned_runtime_is_unread_not_ok() {
    with_gpu(|rt| {
        let mut rng = Rng(0x9015);
        let fault = rt.alloc_buffer(4).expect("fault");
        fault.try_zero().expect("zero");
        dispatch_and_sync(rt, &Case::random(&mut rng, 8, 3, 64, 64), &fault);
        assert_eq!(
            check_dequant_affine_fault(&fault),
            Ok(()),
            "healthy runtime, clean word"
        );
        rt.poison_as_shared_event_timeout_for_test();
        // The word's bytes are still 0, but the host cannot be sure of that.
        match check_dequant_affine_fault(&fault) {
            Err(DequantAffineFault::Unread { why }) => assert!(!why.is_empty()),
            other => panic!("a poisoned runtime must give Unread, not {other:?}"),
        }
    });
}

#[test]
fn check_refuses_a_short_buffer_and_reads_a_six_byte_one() {
    with_gpu(|rt| {
        let mut rng = Rng(0x6b);
        let short = rt.alloc_buffer(2).expect("2 bytes");
        assert_eq!(short.nbytes(), 2);
        match check_dequant_affine_fault(&short) {
            Err(DequantAffineFault::Unread { why }) => assert!(why.contains("2 bytes"), "{why}"),
            other => panic!("a 2-byte buffer must give Unread, not {other:?}"),
        }

        // Not a multiple of four: a `u32` mapping would refuse it, a byte read does not.
        let six = rt.alloc_buffer(6).expect("6 bytes");
        assert_eq!(
            six.nbytes(),
            6,
            "alloc_buffer(6) must be 6 bytes for this test to mean anything"
        );
        six.try_zero().expect("zero");
        // The dispatch path accepts it too: it needs 4 bytes, not a multiple of 4.
        dispatch_and_sync(rt, &Case::random(&mut rng, 8, 3, 64, 64), &six);
        assert_eq!(check_dequant_affine_fault(&six), Ok(()));
        dispatch_and_sync(rt, &overflowing(&mut rng, 8), &six);
        assert_eq!(
            check_dequant_affine_fault(&six),
            Err(DequantAffineFault::Raised { word: 1 })
        );
    });
}
