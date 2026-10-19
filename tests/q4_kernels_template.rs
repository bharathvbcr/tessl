//! Bit-exact pin on the simdgroup MLX Q4 GEMV / GEMM kernels and the util
//! copy / cast kernels.
//!
//! * `gemv_q4_mlx.metal`: `gemv_q4_mlx_simd`, `_simd_add`, `_simd_gate_up_gelu`,
//!   `_simd_kv`, `_simd_qkv` and their `_i4` twins.
//! * `gemm_q4_mlx.metal`: `gemm_q4_mlx_simd`, `_simd_add`, `_simd_i4`,
//!   `_simd_add_i4`.
//! * `utils.metal`: `copy_{f32,u32,bf16,f16}`, `cast_{f32_to_bf16,bf16_to_f32,
//!   f32_to_f16,f16_to_f32}`.
//!
//! Each kernel runs on fixed seeded inputs (odd row counts, K tails that leave
//! upper lanes idle, every epilogue, both weight layouts; casts over NaN with
//! payloads, infinities, signed zeros, subnormals, round-to-even ties and the
//! exhaustive set of 16-bit patterns). An FNV-1a hash of every output word
//! (including the sentinel-filled tail, so a stray write moves it) and the first
//! eight words are compared with values recorded from the kernels *before* they
//! were folded into shared templates. A tolerance test against an f64 reference
//! would not notice a reordered sum; this does.

mod common;

use common::{buf_bf16, with_gpu};
use std::sync::Arc;
use tessl::dispatch::{dispatch_1d, set_gpu_buf, set_u32};
use tessl::nn::{self, GateUpDispatch, Q4MlxBank, Q4MlxLayout, QkvOutputs, QuantShape};
use tessl::tensor::{f32_slice_to_bf16, GpuBuffer};
use tessl::GpuRuntime;

const SENTINEL: u32 = 0xA5A5_A5A5;
const SAMPLE: usize = 8;
const PAD_WORDS: usize = 8;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn word(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / 8_388_608.0 - 1.0
    }
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

fn sample(words: &[u32]) -> [u32; SAMPLE] {
    let mut s = [0u32; SAMPLE];
    for (dst, src) in s.iter_mut().zip(words) {
        *dst = *src;
    }
    s
}

type Row = (String, String, u64, [u32; SAMPLE]);

fn row(kernel: &str, case: &str, words: &[u32]) -> Row {
    (kernel.to_string(), case.to_string(), fnv1a(words), sample(words))
}

fn bytes_buf(rt: &Arc<GpuRuntime>, bytes: &[u8]) -> GpuBuffer {
    let b = rt.alloc_buffer(bytes.len().max(4)).unwrap();
    b.write_bytes(bytes);
    b
}

fn f32_buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    bytes_buf(rt, &bytes)
}

/// `words` f32-sized words, all the sentinel.
fn sentinel_buf(rt: &Arc<GpuRuntime>, words: usize) -> GpuBuffer {
    let bytes: Vec<u8> = (0..words).flat_map(|_| SENTINEL.to_le_bytes()).collect();
    bytes_buf(rt, &bytes)
}

fn read_words(rt: &Arc<GpuRuntime>, b: &GpuBuffer, words: usize) -> Vec<u32> {
    rt.synchronize().unwrap();
    b.read_u32()[..words].to_vec()
}

fn shape(rows: usize, cols: usize, group: usize) -> QuantShape {
    QuantShape {
        rows: rows as u32,
        cols: cols as u32,
        group_size: group as u32,
    }
}

fn layout_suffix(layout: Q4MlxLayout) -> &'static str {
    match layout {
        Q4MlxLayout::RowMajor => "",
        Q4MlxLayout::Interleaved4 => "_i4",
    }
}

/// A seeded random bank: arbitrary nibbles and arbitrary finite bf16
/// scale/bias pairs. Interleaved4 banks store whole 4-row tiles.
fn rand_bank(
    rt: &Arc<GpuRuntime>,
    rows: usize,
    cols: usize,
    group: usize,
    layout: Q4MlxLayout,
    seed: u64,
) -> (GpuBuffer, GpuBuffer) {
    let stored = match layout {
        Q4MlxLayout::RowMajor => rows,
        Q4MlxLayout::Interleaved4 => rows.div_ceil(4) * 4,
    };
    let mut rng = Rng::new(seed);
    let packed: Vec<u8> = (0..stored * cols / 2).map(|_| rng.word() as u8).collect();
    let sb: Vec<f32> = (0..stored * (cols / group))
        .flat_map(|_| {
            let scale = 0.002 + 0.05 * rng.unit().abs();
            let bias = 0.5 * rng.unit();
            [scale, bias]
        })
        .collect();
    let p = bytes_buf(rt, &packed);
    let s = rt.alloc_buffer(sb.len().max(1) * 2).unwrap();
    s.write_bf16_bits(&f32_slice_to_bf16(&sb));
    (p, s)
}

fn rand_x(rt: &Arc<GpuRuntime>, n: usize, seed: u64) -> GpuBuffer {
    let mut rng = Rng::new(seed);
    let x: Vec<f32> = (0..n).map(|_| rng.unit() * 2.0).collect();
    buf_bf16(rt, &x)
}

fn rand_f32(rt: &Arc<GpuRuntime>, n: usize, seed: u64) -> GpuBuffer {
    let mut rng = Rng::new(seed);
    let x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
    f32_buf(rt, &x)
}

/// `(case, rows, cols, group)`: odd and tile-straddling row counts, a K tail
/// that leaves upper lanes idle (544, 96), a multi-block K with a tail (2112),
/// and group sizes 32 / 64 / 128.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("r37c544g32", 37, 544, 32),
    ("r64c1024g64", 64, 1024, 64),
    ("r13c96g32", 13, 96, 32),
    ("r6c512g128", 6, 512, 128),
    ("r9c2112g64", 9, 2112, 64),
];

const LAYOUTS: [Q4MlxLayout; 2] = [Q4MlxLayout::RowMajor, Q4MlxLayout::Interleaved4];

fn gemv_rows(rt: &Arc<GpuRuntime>) -> Vec<Row> {
    let mut out = Vec::new();
    for &(case, rows, cols, group) in SHAPES {
        let sh = shape(rows, cols, group);
        let x = rand_x(rt, cols, 0x4100 + cols as u64);
        let resid = rand_f32(rt, rows, 0x4200 + rows as u64);
        for layout in LAYOUTS {
            let sfx = layout_suffix(layout);
            let (p, s) = rand_bank(rt, rows, cols, group, layout, 0x4300 + rows as u64);
            let bank = Q4MlxBank {
                packed: &p,
                scales_biases: &s,
            };

            for with_resid in [false, true] {
                let y = sentinel_buf(rt, rows + PAD_WORDS);
                nn::gemv_q4_mlx_simd(rt, bank, &x, &y, sh, layout, with_resid.then_some(&resid)).unwrap();
                let name = format!("gemv_q4_mlx_simd{}{sfx}", if with_resid { "_add" } else { "" });
                out.push(row(&name, case, &read_words(rt, &y, rows + PAD_WORDS)));
            }

            let (gp, gs) = rand_bank(rt, rows, cols, group, layout, 0x4400 + rows as u64);
            let (up, us) = rand_bank(rt, rows, cols, group, layout, 0x4500 + rows as u64);
            for mid_bf16 in [false, true] {
                let mid = sentinel_buf(rt, rows + PAD_WORDS);
                nn::gemv_q4_mlx_gate_up_gelu(
                    rt,
                    Q4MlxBank {
                        packed: &gp,
                        scales_biases: &gs,
                    },
                    Q4MlxBank {
                        packed: &up,
                        scales_biases: &us,
                    },
                    &x,
                    &mid,
                    sh,
                    GateUpDispatch::Simd(layout),
                    mid_bf16,
                )
                .unwrap();
                let name = format!("gemv_q4_mlx_simd_gate_up_gelu{sfx}");
                let case = format!("{case}-{}", if mid_bf16 { "mid-bf16" } else { "mid-f32" });
                out.push(row(&name, &case, &read_words(rt, &mid, rows + PAD_WORDS)));
            }

            let kb = Q4MlxBank {
                packed: &gp,
                scales_biases: &gs,
            };
            let vb = Q4MlxBank {
                packed: &up,
                scales_biases: &us,
            };
            let ko = sentinel_buf(rt, rows + PAD_WORDS);
            let vo = sentinel_buf(rt, rows + PAD_WORDS);
            nn::gemv_q4_mlx_kv(rt, kb, vb, &x, &ko, &vo, sh, layout).unwrap();
            let mut words = read_words(rt, &ko, rows + PAD_WORDS);
            words.extend(read_words(rt, &vo, rows + PAD_WORDS));
            out.push(row(&format!("gemv_q4_mlx_simd_kv{sfx}"), case, &words));

            let rows_kv = rows / 3 + 1;
            let (qp, qs) = rand_bank(rt, rows, cols, group, layout, 0x4600 + rows as u64);
            let (kp, ks) = rand_bank(rt, rows_kv, cols, group, layout, 0x4700 + rows as u64);
            let (vp, vs) = rand_bank(rt, rows_kv, cols, group, layout, 0x4800 + rows as u64);
            let qo = sentinel_buf(rt, rows + PAD_WORDS);
            let ko = sentinel_buf(rt, rows_kv + PAD_WORDS);
            let vo = sentinel_buf(rt, rows_kv + PAD_WORDS);
            nn::gemv_q4_mlx_qkv(
                rt,
                Q4MlxBank {
                    packed: &qp,
                    scales_biases: &qs,
                },
                Q4MlxBank {
                    packed: &kp,
                    scales_biases: &ks,
                },
                Q4MlxBank {
                    packed: &vp,
                    scales_biases: &vs,
                },
                &x,
                QkvOutputs {
                    q_out: &qo,
                    k_out: &ko,
                    v_out: &vo,
                },
                rows as u32,
                rows_kv as u32,
                cols as u32,
                group as u32,
                layout,
            )
            .unwrap();
            let mut words = read_words(rt, &qo, rows + PAD_WORDS);
            words.extend(read_words(rt, &ko, rows_kv + PAD_WORDS));
            words.extend(read_words(rt, &vo, rows_kv + PAD_WORDS));
            out.push(row(&format!("gemv_q4_mlx_simd_qkv{sfx}"), case, &words));
        }
    }
    out
}

fn gemm_rows(rt: &Arc<GpuRuntime>) -> Vec<Row> {
    let mut out = Vec::new();
    for &(case, rows, cols, group) in &SHAPES[..4] {
        let sh = shape(rows, cols, group);
        for m in [1usize, 5, 8] {
            let x = rand_x(rt, m * cols, 0x5100 + (cols * m) as u64);
            let resid = rand_f32(rt, m * rows, 0x5200 + (rows * m) as u64);
            for layout in LAYOUTS {
                let (p, s) = rand_bank(rt, rows, cols, group, layout, 0x5300 + rows as u64);
                let bank = Q4MlxBank {
                    packed: &p,
                    scales_biases: &s,
                };
                for with_resid in [false, true] {
                    let y = sentinel_buf(rt, m * rows + PAD_WORDS);
                    nn::gemm_q4_mlx(rt, bank, &x, &y, sh, m as u32, layout, with_resid.then_some(&resid)).unwrap();
                    let name = format!(
                        "gemm_q4_mlx_simd{}{}",
                        if with_resid { "_add" } else { "" },
                        layout_suffix(layout)
                    );
                    out.push(row(
                        &name,
                        &format!("{case}-m{m}"),
                        &read_words(rt, &y, m * rows + PAD_WORDS),
                    ));
                }
            }
        }
    }
    out
}

/// f32 bit patterns that decide a narrowing cast: NaN with payloads, infinities,
/// signed zeros, subnormals, the bf16 and f16 round-to-even ties and their
/// neighbours, and the f16 overflow / underflow edges.
const F32_SPECIALS: &[u32] = &[
    0x7fc0_0000,
    0x7f80_0001,
    0xffc1_2345,
    0xff80_0001,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x0080_0000,
    0x3f80_8000,
    0x3f81_8000,
    0x3f80_7fff,
    0x3f80_8001,
    0xbf80_8000,
    0xbf81_8000,
    0x7f7f_ffff,
    0x7f7f_8000,
    0xff7f_ffff,
    0x477f_e000,
    0x477f_ef00,
    0x477f_f000,
    0xc77f_f000,
    0x4780_0000,
    0x3380_0000,
    0x3300_0000,
    0x3340_0000,
    0x3300_0001,
    0x33c0_0000,
    0x3f80_1000,
    0x3f80_3000,
    0x3880_0000,
    0x387f_e000,
    0x387f_ff00,
    0xb300_0000,
    0xb3c0_0000,
];

fn f32_source_words(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| match F32_SPECIALS.get(i) {
            Some(&w) => w,
            None => rng.word(),
        })
        .collect()
}

/// Every 16-bit pattern once, then `n - 65536` more.
fn pattern16(n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| (i as u32).wrapping_mul(0x9e37) as u16 ^ (i >> 16) as u16)
        .collect()
}

fn le_bytes_u32(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn le_bytes_u16(w: &[u16]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One elementwise kernel over `n` elements. The output buffer is sentinel
/// filled and read back whole (rounded up to a word), so the unwritten half of
/// the last word of an odd 16-bit output is part of the hash.
fn run_elementwise(rt: &Arc<GpuRuntime>, kernel: &str, input: &[u8], n: usize, out_elem_bytes: usize) -> Vec<u32> {
    let out_words = (n * out_elem_bytes).div_ceil(4);
    let inb = bytes_buf(rt, input);
    let outb = sentinel_buf(rt, out_words);
    let p = rt.pipeline(kernel).unwrap();
    dispatch_1d(rt, &p, n, |bnd| {
        set_gpu_buf(bnd, &inb, 0);
        set_gpu_buf(bnd, &outb, 1);
        set_u32(bnd, n as u32, 2);
    })
    .unwrap();
    read_words(rt, &outb, out_words)
}

fn util_rows(rt: &Arc<GpuRuntime>) -> Vec<Row> {
    let mut out = Vec::new();
    // 4-byte-source kernels.
    for (kernel, out_bytes) in [
        ("copy_f32", 4usize),
        ("copy_u32", 4),
        ("cast_f32_to_bf16", 2),
        ("cast_f32_to_f16", 2),
    ] {
        for n in [4099usize, 1, 33] {
            let src = f32_source_words(n, 0x6100 + n as u64);
            let w = run_elementwise(rt, kernel, &le_bytes_u32(&src), n, out_bytes);
            out.push(row(kernel, &format!("n{n}"), &w));
        }
    }
    // 2-byte-source kernels: every 16-bit pattern, plus odd counts.
    for (kernel, out_bytes) in [
        ("copy_bf16", 2usize),
        ("copy_f16", 2),
        ("cast_bf16_to_f32", 4),
        ("cast_f16_to_f32", 4),
    ] {
        for n in [65537usize, 1, 33] {
            let src = pattern16(n);
            let w = run_elementwise(rt, kernel, &le_bytes_u16(&src), n, out_bytes);
            out.push(row(kernel, &format!("n{n}"), &w));
        }
    }
    out
}

fn all_rows(rt: &Arc<GpuRuntime>) -> Vec<Row> {
    let mut v = gemv_rows(rt);
    v.extend(gemm_rows(rt));
    v.extend(util_rows(rt));
    v
}

type Golden = (&'static str, &'static str, u64, [u32; SAMPLE]);

const GOLDEN: &[Golden] = &[
    (
        "gemv_q4_mlx_simd",
        "r37c544g32",
        0x9bf0e28351f1a0ee,
        [
            0xc0696a5a, 0x40e3dde0, 0xc0c18428, 0xc1780175, 0xc0ed23a4, 0x3fa40378, 0x3ff8b6a0, 0xc064fd70,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add",
        "r37c544g32",
        0x22995a778616a574,
        [
            0xc0742226, 0x40d2d7d4, 0xc0bebb6b, 0xc17c4083, 0xc10590ce, 0x4004b2cc, 0x400476a3, 0xc03fbe5c,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r37c544g32-mid-f32",
        0xfbf4524f9b6ce277,
        [
            0x80000000, 0xbf29a18e, 0xc2694d3e, 0xc20797c8, 0x80000000, 0x00000000, 0xc1cbe7f2, 0xc16b6610,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r37c544g32-mid-bf16",
        0xe1ad2b227a0260ca,
        [
            0xbf2a8000, 0xc208c269, 0x00008000, 0xc16bc1cc, 0x423d403c, 0x00003f4c, 0x0000beb8, 0x8000b893,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv",
        "r37c544g32",
        0x6b7fa77509438587,
        [
            0xc12defc5, 0xbfdc8690, 0x40b7ec1a, 0x4133dd67, 0xc197c893, 0xc12bafcd, 0x4175a398, 0x40df3a8c,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv",
        "r37c544g32",
        0xdb0ea3d1d51d2963,
        [
            0xc1a9354d, 0x40ed4f68, 0xbfffda80, 0xc16fe7c9, 0x4153a910, 0xc1342aee, 0xc08aa172, 0xc1806680,
        ],
    ),
    (
        "gemv_q4_mlx_simd_i4",
        "r37c544g32",
        0x9ae200633176b21b,
        [
            0x401f1ddb, 0xc1639bee, 0x415bb284, 0xc1aa6ec3, 0xc0e92329, 0xc148ca42, 0x41494cc8, 0xc09eaa33,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add_i4",
        "r37c544g32",
        0xc09094dbc7ef3d87,
        [
            0x40146610, 0xc16c1ef4, 0x415d16e3, 0xc1ac8e4a, 0xc1039091, 0xc13c1dfe, 0x414b539d, 0xc08c0aa9,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r37c544g32-mid-f32",
        0x7ae21d6b91baeffb,
        [
            0xbc7b94f1, 0x3cf500dd, 0x370d3269, 0x00000000, 0xc1ca0317, 0x4273d231, 0x80000000, 0x3f74d8bd,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r37c544g32-mid-bf16",
        0xc242649f8292f350,
        [
            0x3cf5bc7c, 0x0000370d, 0x4274c1ca, 0x3f758000, 0xc0cd8000, 0xc2b84248, 0x00008000, 0x8000bc72,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv_i4",
        "r37c544g32",
        0x01dfc6692a1d6726,
        [
            0xc04867a9, 0xc03ebc16, 0xc09ac9e4, 0xc0adc112, 0x408b70ad, 0x409b09e2, 0xc0ce6932, 0xbf2961e0,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv_i4",
        "r37c544g32",
        0x0dee059983496cc1,
        [
            0xc0a8529c, 0xc135b0e1, 0x4143be32, 0xc103d29d, 0xc0d4bd7d, 0x4112e28e, 0xc08a99b3, 0xc11e4ce5,
        ],
    ),
    (
        "gemv_q4_mlx_simd",
        "r64c1024g64",
        0x8348f7bf7330db4f,
        [
            0x4197c8ff, 0x41a5e3ff, 0x41b9f54a, 0xc13138bc, 0x40acab4a, 0x41d2a76c, 0x3f550af0, 0x4231abdc,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add",
        "r64c1024g64",
        0x82b581a85330c99f,
        [
            0x419dc20c, 0x41a9b675, 0x41baa89c, 0xc13618e0, 0x40b10d2e, 0x41d09299, 0x3e3ee890, 0x4231b043,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r64c1024g64-mid-f32",
        0xaaf2d0e23c239481,
        [
            0x432a34d7, 0xc1da1047, 0x00000000, 0x43e74de9, 0x44726a42, 0x42c7f329, 0xc17e7ee5, 0xb905d392,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r64c1024g64-mid-bf16",
        0xc1e9b7849b2e8367,
        [
            0xc1da432a, 0x43e70000, 0x42c84472, 0xb906c17e, 0x42e3c401, 0x438742fe, 0x424cc407, 0x42e34433,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv",
        "r64c1024g64",
        0x5ddf8618463ca22d,
        [
            0x41bc3053, 0x40f41f38, 0xc1910f7f, 0x41b1e5ad, 0x41ca8968, 0x41db0c09, 0x418d55ac, 0xc08b30dc,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv",
        "r64c1024g64",
        0x02156f8a5fd4bb79,
        [
            0x4197a75f, 0x41b4e72b, 0x40efa5e8, 0x406d2422, 0x4101dd28, 0x41bdae20, 0x41cd40f4, 0x413c8afe,
        ],
    ),
    (
        "gemv_q4_mlx_simd_i4",
        "r64c1024g64",
        0x8d1b40c711fdb515,
        [
            0xc0b9c38c, 0x409de167, 0x3f3544e4, 0xc1792a29, 0x3f309d48, 0x41d39019, 0xc0ec571a, 0x41ba1005,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add_i4",
        "r64c1024g64",
        0x33bad27ecf9d15f2,
        [
            0xc0a1df5a, 0x40ad2b40, 0x3f4baf2e, 0xc17e0a4e, 0x3f53ac6c, 0x41d17b46, 0xc100809a, 0x41ba18d3,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r64c1024g64-mid-f32",
        0xf746d01da7cf88d2,
        [
            0x00000000, 0x4394b2eb, 0x43a0a112, 0x449eef75, 0xc35bda9e, 0xc1634ec6, 0x4270da5b, 0x3db7a694,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r64c1024g64-mid-bf16",
        0xe443f05fded4f48a,
        [
            0x43950000, 0x449f43a1, 0xc163c35c, 0x3db84271, 0x43313f71, 0x3a73419f, 0xc0d142f5, 0x00004214,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv_i4",
        "r64c1024g64",
        0x6b99889126a1894b,
        [
            0xc1318dc6, 0x4128a000, 0x4128e825, 0x422385cc, 0x42131bda, 0x41407b39, 0x409b4394, 0xc00711f8,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv_i4",
        "r64c1024g64",
        0xd75641bf3dd54750,
        [
            0x40be5c60, 0x4165fdaf, 0x413cf696, 0xc1234fba, 0xc1348aa7, 0x4186b309, 0x401710e8, 0xc1453526,
        ],
    ),
    (
        "gemv_q4_mlx_simd",
        "r13c96g32",
        0x1d0a93f65f8c1892,
        [
            0x3f43f7fb, 0xbea6c54c, 0xc00414f4, 0x3fc807a2, 0x3f5285ea, 0xbfe514ee, 0xbe2547f8, 0xbe01e7a0,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add",
        "r13c96g32",
        0x94afe2d5636093a2,
        [
            0xbe666c54, 0x3f17eea6, 0xbfd457c4, 0x3fa3df5a, 0x3eea93c0, 0xc0104fa4, 0xbf68d53c, 0x3d18c060,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r13c96g32-mid-f32",
        0x8e87173137dfca36,
        [
            0x3e1fe802, 0xbf033f04, 0xbd29ce3d, 0x3e7fffb0, 0x3bb98a21, 0x3d015343, 0x4028b9e0, 0xbd501899,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r13c96g32-mid-bf16",
        0xd65dcf921480491d,
        [
            0xbf033e20, 0x3e80bd2a, 0x3d013bba, 0xbd504029, 0xbfc5b9b2, 0xbfe48000, 0xa5a5bdb1, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv",
        "r13c96g32",
        0x670dd9bc80525af7,
        [
            0xbfe79722, 0xbef36850, 0xbff74425, 0xbe904c78, 0xbfe63768, 0xc0013904, 0x3ff0b596, 0x3d4ee380,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv",
        "r13c96g32",
        0x395fc02f0ad2b19d,
        [
            0xbfb0266e, 0xbf846bc8, 0xc002cfed, 0xc03d4e98, 0xc03dbe55, 0xbf93f858, 0xc03f0cce, 0x3f353717,
        ],
    ),
    (
        "gemv_q4_mlx_simd_i4",
        "r13c96g32",
        0xc9320c2a7453373f,
        [
            0xc0068d80, 0x3f8c100d, 0xc082e485, 0x3e870316, 0xbf9d39f6, 0xbf82519f, 0x3f3099ea, 0xbf19d255,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add_i4",
        "r13c96g32",
        0xf329d42a3f038fbc,
        [
            0xc045f244, 0x4000dc5a, 0xc06bdff8, 0xbc99e0a0, 0xbfcbd7fb, 0xbfbddbf9, 0xbd6e9540, 0xbedf98ce,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r13c96g32-mid-f32",
        0x3d86830049f851c1,
        [
            0x3efe4774, 0xbda46128, 0x3c54f95b, 0x3acfb74e, 0x3c0f6db4, 0xbf2f0507, 0x3dcf3357, 0x3d2092f0,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r13c96g32-mid-bf16",
        0x6367b6a8039b83e0,
        [
            0xbda43efe, 0x3ad03c55, 0xbf2f3c0f, 0x3d213dcf, 0x3b74bd42, 0xbafbba94, 0xa5a5401f, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv_i4",
        "r13c96g32",
        0x866f9ad1b5a2e269,
        [
            0xbf3e21f0, 0xbf2d4cb4, 0xc0064c40, 0xc06465bf, 0xbd36bd90, 0x3facf4f6, 0xbfeebfc6, 0x3d918ff0,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv_i4",
        "r13c96g32",
        0xde562da32501300d,
        [
            0xbf35b7ae, 0x3fde3726, 0x3ab4a000, 0xbfb1b429, 0xc02d2106, 0xbfc4ec29, 0xc07a665a, 0xc089eeb0,
        ],
    ),
    (
        "gemv_q4_mlx_simd",
        "r6c512g128",
        0x375756b60d0d831e,
        [
            0xc15edecc, 0xc16beea2, 0xc1140bcc, 0x40eb9bd7, 0xc01ce2f2, 0x3eca9940, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add",
        "r6c512g128",
        0x885cda8027462633,
        [
            0xc15dbca0, 0xc163536c, 0xc10a2c80, 0x410413d3, 0xc03d267a, 0x3f2c557e, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r6c512g128-mid-f32",
        0x52ff9eb871d32225,
        [
            0xc260b506, 0x00000000, 0x00000000, 0x39c5bbbf, 0x3dbc61f2, 0xbee1e3c3, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r6c512g128-mid-bf16",
        0xc621dd559efbf868,
        [
            0x0000c261, 0x39c60000, 0xbee23dbc, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv",
        "r6c512g128",
        0x8270850a396eec1a,
        [
            0x4106d5e3, 0xc0fa2299, 0xc0a1b8d6, 0xc07ccb46, 0x3f6f33c0, 0xc00ffef6, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv",
        "r6c512g128",
        0xde155ed7bfa45b05,
        [
            0x400f4c44, 0xc0296150, 0xc0ea7fe0, 0xc180338c, 0x4011adb3, 0x4155d33d, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_i4",
        "r6c512g128",
        0x466db9337d6b80e1,
        [
            0x403efc52, 0xbf8c9de8, 0xbffa8aca, 0xbfdd4cd0, 0xc11dd2a3, 0x414f824c, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add_i4",
        "r6c512g128",
        0xfb2f0bc9efe72868,
        [
            0x40438501, 0xbf0f886c, 0xbfab9066, 0xbf563b22, 0xc125e385, 0x4153f2da, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r6c512g128-mid-f32",
        0x49b1d9624a4d93d2,
        [
            0x40358b7f, 0x42781a0b, 0x41012575, 0x80000000, 0xc371f68e, 0x3ea41c2d, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r6c512g128-mid-bf16",
        0xa1d804e1a99cec22,
        [
            0x42784036, 0x80004101, 0x3ea4c372, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv_i4",
        "r6c512g128",
        0x38f3d8e4e4996a30,
        [
            0xbf5abd2c, 0x40bc40ad, 0x3fb33f30, 0xc19958a7, 0x4170d939, 0xbff5eaf8, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv_i4",
        "r6c512g128",
        0x00782168e8960eea,
        [
            0x40262328, 0xc088f1a9, 0x414ee5fe, 0x3f8779c8, 0xc0ef0cfe, 0xc1967153, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd",
        "r9c2112g64",
        0x459e479c791c907f,
        [
            0x409aae32, 0xc1154e3d, 0xc1a0fded, 0x40c68abe, 0x41565026, 0xc22c9df2, 0xc20cbd32, 0x408ac128,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add",
        "r9c2112g64",
        0x94e0ec803638ed6f,
        [
            0x409b8c40, 0xc10a21e6, 0xc19a0305, 0x40c8fe94, 0x415b652d, 0xc22aeaf6, 0xc2098518, 0x4096abb3,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r9c2112g64-mid-f32",
        0xd2c3552b1c5f7430,
        [
            0x80000000, 0xbfa1d31f, 0x80000000, 0x00000000, 0x00000000, 0xb84eca2c, 0xc0b87fb5, 0xc2bf6268,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu",
        "r9c2112g64-mid-bf16",
        0xadadad20e3c5a73e,
        [
            0xbfa28000, 0x00008000, 0xb84f0000, 0xc2bfc0b8, 0xa5a50000, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv",
        "r9c2112g64",
        0x7b2c22793c60c81f,
        [
            0xc1280b58, 0xbfd9e838, 0xc1e4aa0b, 0xc1cb7309, 0xc19427dc, 0xc0944c08, 0x415acf8c, 0x40c61cda,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv",
        "r9c2112g64",
        0xd5fd08673b1c08ea,
        [
            0xc23feb2e, 0xc18f4252, 0xc017e10e, 0xc23b7f16, 0x40c701f3, 0xc223b7d1, 0xc1a2466c, 0xc01b6756,
        ],
    ),
    (
        "gemv_q4_mlx_simd_i4",
        "r9c2112g64",
        0x236caf8929236503,
        [
            0xc1943ff3, 0xc10d69fc, 0xc154d6bc, 0xc0db10b0, 0xbfb87ba0, 0xc150a521, 0xc0832598, 0xc21b3abf,
        ],
    ),
    (
        "gemv_q4_mlx_simd_add_i4",
        "r9c2112g64",
        0x31f335e34bf63100,
        [
            0xc194086f, 0xc1023da5, 0xc146e0ec, 0xc0d89cda, 0xbf8fd368, 0xc149d930, 0xc052c989, 0xc219bd6e,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r9c2112g64-mid-f32",
        0x3dda40714b30a7e5,
        [
            0x00000000, 0xc3aff4e8, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0xc363e5f3, 0x80000000,
        ],
    ),
    (
        "gemv_q4_mlx_simd_gate_up_gelu_i4",
        "r9c2112g64-mid-bf16",
        0x99fbe794e060078b,
        [
            0xc3b00000, 0x00008000, 0x00000000, 0x8000c364, 0xa5a50000, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemv_q4_mlx_simd_kv_i4",
        "r9c2112g64",
        0x7e606837510310b7,
        [
            0xc14b7496, 0x412a5b13, 0xc1bb00c1, 0xc0d3cad6, 0xc1f5a99e, 0xc1303545, 0x41bbfc4c, 0xc1300c88,
        ],
    ),
    (
        "gemv_q4_mlx_simd_qkv_i4",
        "r9c2112g64",
        0x4df41ce1559850df,
        [
            0xc2503c9b, 0xbfcd9d26, 0xc176cfe8, 0xc0a83396, 0x413821f0, 0x40c799b1, 0x411f7d0e, 0x40b19e71,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r37c544g32-m1",
        0x6eb97d69e4eb971e,
        [
            0x3e146dd0, 0x41510956, 0xc0d266f5, 0x406d91be, 0x413d8500, 0x3f5340a0, 0xc06cd153, 0x41561dd6,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r37c544g32-m1",
        0x25d747e60136362d,
        [
            0x3f8a0c04, 0x415ec99a, 0xc0b4f73c, 0x406daa70, 0x4139ac20, 0x3f2a4312, 0xc04f849c, 0x4163f060,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r37c544g32-m1",
        0x45f2604351ac5423,
        [
            0x410e5812, 0x3e7b3020, 0xc0330f38, 0x40f13e17, 0xc011e5e9, 0x405ce201, 0x405b0f5d, 0xbf039320,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r37c544g32-m1",
        0xce7d51b9244d718b,
        [
            0x411d47db, 0x3f8d6821, 0xbff05f8c, 0x40f14a70, 0xc0214968, 0x4052a29e, 0x40785c14, 0x3eb32af0,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r37c544g32-m5",
        0x449c0515d8ac96b8,
        [
            0xbda65030, 0xc1391d61, 0x400122a8, 0xc185ead8, 0x407ddfde, 0x4118be01, 0xc10b1e38, 0xc0c0f09f,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r37c544g32-m5",
        0x2b45ca1442741dbe,
        [
            0x3edb65b0, 0xc132ca5c, 0x3fe2eb56, 0xc185eafe, 0x405d4c00, 0x411e8971, 0xc10cafb0, 0xc0da0933,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r37c544g32-m5",
        0xae8551ca1ee1f737,
        [
            0xc0c59f30, 0x4106c133, 0xc13e31ae, 0xc1c912ca, 0xc139b05e, 0x40ec6498, 0xc100f665, 0x40bde457,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r37c544g32-m5",
        0x3232925f6e94df45,
        [
            0xc0b54f94, 0x410d1438, 0xc1421ced, 0xc1c912f0, 0xc141d556, 0x40f7fb78, 0xc10287dd, 0x40a4cbc3,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r37c544g32-m8",
        0xbcd6d4041c7b6c59,
        [
            0xc0ab123b, 0xc09e8643, 0x40a8f95d, 0xc03af362, 0xc0d3297b, 0xc03dd865, 0xc118f0d6, 0x40388b0c,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r37c544g32-m8",
        0x28e3278209b4ac97,
        [
            0xc0a90214, 0xc0865a1e, 0x40b7b3e3, 0xc024137f, 0xc0b45fe9, 0xc01e73f0, 0xc11ee925, 0x404e8be6,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r37c544g32-m8",
        0x575654be93a3bacf,
        [
            0x401bcbec, 0xbfe83722, 0x400f7296, 0x40638641, 0x410f8227, 0xc03601f5, 0xc193e5ad, 0xc0c9a5d4,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r37c544g32-m8",
        0x1ad1bf5c38380b53,
        [
            0x401fec3a, 0xbf878690, 0x402ce7a2, 0x407a6624, 0x411ee6f0, 0xc0169d80, 0xc196e1d4, 0xc0bea567,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r64c1024g64-m1",
        0x07350398270b3c24,
        [
            0xc0031e28, 0xc197a548, 0xc21989f8, 0x3f16afa0, 0x40037f74, 0xc0f29eb0, 0xc1bea924, 0xc0c44bd8,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r64c1024g64-m1",
        0xd4fac330d6f7b869,
        [
            0xc018aa56, 0xc1915ec4, 0xc21d4cdd, 0x3fbeef76, 0x3fd29d6e, 0xc0e9217f, 0xc1c5207b, 0xc0cd7860,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r64c1024g64-m1",
        0x97e8a527604ad9fe,
        [
            0xc1b0fc05, 0xc1d53056, 0xc03a2fcc, 0xc144c1d8, 0xc1ed5dd4, 0xc1f6b73a, 0xc18f862d, 0xbf896498,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r64c1024g64-m1",
        0xe7d5e87a7b470b46,
        [
            0xc1b3ad8b, 0xc1cee9d2, 0xc0765e22, 0xc1364ee3, 0xc1f0a3ec, 0xc1f457ee, 0xc195fd84, 0xbfae16b8,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r64c1024g64-m5",
        0x809630017f3d1dc9,
        [
            0x40a0f5d3, 0xc17c3c1a, 0x404633f2, 0xc0af2f44, 0xc06af2c0, 0x40083544, 0xbfa70691, 0x3dd61c00,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r64c1024g64-m5",
        0x29313027c54162a0,
        [
            0x40af5c08, 0xc184e4b2, 0x404af122, 0xc0b13f70, 0xc03bd649, 0x403ee4f8, 0xbff8e866, 0x3e311cf8,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r64c1024g64-m5",
        0x6a37bb79d4d04ca5,
        [
            0xc0ac61f6, 0xbe622480, 0x4122fcae, 0x419146b2, 0xc0702407, 0x40e72860, 0xc1948a5b, 0x416b4569,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r64c1024g64-m5",
        0x8e5d63942b9fcd0b,
        [
            0xc09dfbc1, 0xbf88aee8, 0x41242bfa, 0x4190c2a7, 0xc0410790, 0x4101401d, 0xc199a878, 0x416c5da5,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r64c1024g64-m8",
        0xb19ee8caaf70a610,
        [
            0xc08ef149, 0xbdd59940, 0x41ab3ae6, 0xc20742fa, 0xc1484fc5, 0x4182ecb3, 0xc1713240, 0xc0f83012,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r64c1024g64-m8",
        0xbb9a04d5ec7848c9,
        [
            0xc0a8be60, 0xbf311ba4, 0x41b0710a, 0xc204d3e2, 0xc14255e3, 0x41822b45, 0xc17ae85c, 0xc10624cf,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r64c1024g64-m8",
        0xa83d5f805904760d,
        [
            0xc0a52f06, 0x3fabeff0, 0xc14e09c4, 0xc15beb20, 0x41408fbe, 0xc1abeeb2, 0xc0a046be, 0xc1b1163c,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r64c1024g64-m8",
        0x734cd2ddf645835d,
        [
            0xc0befc1d, 0x3f417764, 0xc1439d7c, 0xc1522ebf, 0x414689a0, 0xc1acb020, 0xc0b3b2f5, 0xc1b61c9f,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r13c96g32-m1",
        0x72d86b5c1e4063e0,
        [
            0x40bba7ac, 0xc088b37c, 0xbfe896fe, 0x3f88c82c, 0x407cd70d, 0x4084aa6e, 0xbfffd1d4, 0x400be8e4,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r13c96g32-m1",
        0x63907a4a50127bad,
        [
            0x40d69265, 0xc0825f2d, 0xc029eaf6, 0x3e9b2348, 0x408dc2b2, 0x4097b6fc, 0xc02b3420, 0x3ff0c02f,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r13c96g32-m1",
        0xac8ed663acc343a3,
        [
            0x407cc734, 0x402a6230, 0x3ecc8f40, 0x40212da6, 0x3ec05a2c, 0xbebd3f40, 0xc05cf916, 0x3f20a290,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r13c96g32-m1",
        0x3167800ebe8d317b,
        [
            0x40994e53, 0x40370ace, 0xbee06c7c, 0x3fe05bf2, 0x3f5ae674, 0x3e671340, 0xc0842226, 0x3ea4febc,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r13c96g32-m5",
        0xd756ff3d065bc1ad,
        [
            0xbf96b9a6, 0x4136a617, 0x40fa8dae, 0x3fcad2f1, 0x40537dab, 0xbff80aab, 0x3fa2aca2, 0x4079fa70,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r13c96g32-m5",
        0xb8b1bce2f605778d,
        [
            0xbfbe9b93, 0x413ccbc4, 0x40df262c, 0x3fb70fb1, 0x405ae4e5, 0xc00abab0, 0x3f768bc4, 0x405557fe,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r13c96g32-m5",
        0xf603ef71278df726,
        [
            0x40cba90c, 0xbfba6b61, 0x3f8e3134, 0xc01fd242, 0x40303d5e, 0xc0455358, 0x40a6da72, 0x3fdd2d70,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r13c96g32-m5",
        0xc7971af56c9b822b,
        [
            0x40c1b091, 0xbf893dfa, 0x3e824cb8, 0xc029b3e2, 0x4037a498, 0xc05408b3, 0x409d00c2, 0x3f93e88b,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r13c96g32-m8",
        0xc1c2fd9c8d323609,
        [
            0x409c111e, 0x3f943120, 0x3fbeb5f3, 0x3feb11b4, 0x40725b94, 0x40b58efd, 0x4022f17d, 0x408626c2,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r13c96g32-m8",
        0x1acd9df6ddb40be2,
        [
            0x4088e3e8, 0x3fbf36e4, 0x3f78bfaa, 0x3ff1bad7, 0x408e027f, 0x40a1d690, 0x3fc8b37d, 0x40761519,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r13c96g32-m8",
        0x0f0ce9248f246fc6,
        [
            0x404e354e, 0x4083c59e, 0x4108e2b2, 0x4091f535, 0x3f237d5c, 0xc03982f9, 0xc08a4962, 0xbfad4ea2,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r13c96g32-m8",
        0x97a9a2d543f570f8,
        [
            0x4027dae2, 0x408e870f, 0x410097ee, 0x40939f7e, 0x3fa51183, 0xc060f3d3, 0xc0a99541, 0xbfd9bf78,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r6c512g128-m1",
        0xd3ee00abd28f56c4,
        [
            0x4017e640, 0x40d3a38f, 0x412fc710, 0xc0eb6b90, 0x40e79827, 0x4162116c, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r6c512g128-m1",
        0x45dba228808206e9,
        [
            0x3fe3b3a8, 0x40db9950, 0x4136c84b, 0xc0e07137, 0x40fdb38b, 0x41661138, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r6c512g128-m1",
        0x65930c5960dd166a,
        [
            0x3ff43614, 0x4036fbb4, 0x3fdae510, 0xc0ad3aca, 0x413bea3f, 0x4085d43e, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r6c512g128-m1",
        0xc552386b60974a4f,
        [
            0x3fa81d3c, 0x4046e736, 0x40097772, 0xc0a24071, 0x4146f7f1, 0x408dd3d5, 0xa5a5a5a5, 0xa5a5a5a5,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r6c512g128-m5",
        0x840b11612103977e,
        [
            0x401a99ac, 0xc0b599e2, 0x3fa4dbec, 0xc157e3a6, 0x41023877, 0x3f0180e0, 0x413c83e4, 0xc09b1338,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r6c512g128-m5",
        0xc0348d5854b1aeba,
        [
            0x402f478b, 0xc0d274d1, 0x3fd06dc3, 0xc1673242, 0x410952d6, 0x3f8e8516, 0x414aad93, 0xc0926c0a,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r6c512g128-m5",
        0x1aca2ea281130e95,
        [
            0xbff6daf8, 0xc0b534ba, 0x40c4d70d, 0xc1b7802e, 0xc0c33e57, 0xc0ede7a1, 0x4153eab5, 0x415ec6d5,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r6c512g128-m5",
        0xe7bd0b5dfa83f5b4,
        [
            0xbfcd7f3a, 0xc0d20fa9, 0x40cfbb83, 0xc1bf277c, 0xc0b5099a, 0xc0da7678, 0x41621464, 0x41631a6c,
        ],
    ),
    (
        "gemm_q4_mlx_simd",
        "r6c512g128-m8",
        0x58ee215a9c1b90db,
        [
            0xc1461092, 0xc11bc711, 0xbfa99152, 0xc09fce5c, 0xc0c2998c, 0xc12621d6, 0xc117941a, 0x40a94e96,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add",
        "r6c512g128-m8",
        0x6c45db1b8b53f1b6,
        [
            0xc1529074, 0xc11060bf, 0xbfa75063, 0xc0bc594b, 0xc0a4e3b8, 0xc126c458, 0xc11cde3c, 0x40b504df,
        ],
    ),
    (
        "gemm_q4_mlx_simd_i4",
        "r6c512g128-m8",
        0xfb7065881e5ba5f0,
        [
            0xc08c85de, 0xc1105c14, 0x3fe9b742, 0xc0fad6c0, 0xc164b113, 0x40919c89, 0xc0de671e, 0xbf8de984,
        ],
    ),
    (
        "gemm_q4_mlx_simd_add_i4",
        "r6c512g128-m8",
        0x19101ede48014afc,
        [
            0xc0a585a3, 0xc104f5c2, 0x3febf831, 0xc10bb0d8, 0xc155d629, 0x40905786, 0xc0e8fb62, 0xbf3e20c0,
        ],
    ),
    (
        "copy_f32",
        "n4099",
        0xf72b83e3941dffd1,
        [
            0x7fc00000, 0x7f800001, 0xffc12345, 0xff800001, 0x7f800000, 0xff800000, 0x00000000, 0x80000000,
        ],
    ),
    (
        "copy_f32",
        "n1",
        0x4a98877f9ba2f898,
        [
            0x7fc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
    ),
    (
        "copy_f32",
        "n33",
        0xed9486a76a568114,
        [
            0x7fc00000, 0x7f800001, 0xffc12345, 0xff800001, 0x7f800000, 0xff800000, 0x00000000, 0x80000000,
        ],
    ),
    (
        "copy_u32",
        "n4099",
        0xf72b83e3941dffd1,
        [
            0x7fc00000, 0x7f800001, 0xffc12345, 0xff800001, 0x7f800000, 0xff800000, 0x00000000, 0x80000000,
        ],
    ),
    (
        "copy_u32",
        "n1",
        0x4a98877f9ba2f898,
        [
            0x7fc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
    ),
    (
        "copy_u32",
        "n33",
        0xed9486a76a568114,
        [
            0x7fc00000, 0x7f800001, 0xffc12345, 0xff800001, 0x7f800000, 0xff800000, 0x00000000, 0x80000000,
        ],
    ),
    (
        "cast_f32_to_bf16",
        "n4099",
        0x20ca9659c67fc8c3,
        [
            0x7fc07fc0, 0x7fc07fc0, 0xff807f80, 0x80000000, 0x80000000, 0x80000000, 0x3f800080, 0x3f803f82,
        ],
    ),
    (
        "cast_f32_to_bf16",
        "n1",
        0x7d8550c18625b0ee,
        [
            0xa5a57fc0, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
    ),
    (
        "cast_f32_to_bf16",
        "n33",
        0x401e8b3912dca76e,
        [
            0x7fc07fc0, 0x7fc07fc0, 0xff807f80, 0x80000000, 0x80000000, 0x80000000, 0x3f800080, 0x3f803f82,
        ],
    ),
    (
        "cast_f32_to_f16",
        "n4099",
        0xbcb2f3c134414ad6,
        [
            0x7e007e00, 0x7e007e00, 0xfc007c00, 0x80000000, 0x80000000, 0x80000000, 0x3c040000, 0x3c043c0c,
        ],
    ),
    (
        "cast_f32_to_f16",
        "n1",
        0x32dd437e6c914605,
        [
            0xa5a57e00, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
    ),
    (
        "cast_f32_to_f16",
        "n33",
        0xc4860567f93ee174,
        [
            0x7e007e00, 0x7e007e00, 0xfc007c00, 0x80000000, 0x80000000, 0x80000000, 0x3c040000, 0x3c043c0c,
        ],
    ),
    (
        "copy_bf16",
        "n65537",
        0x55c8a7c2aa06d882,
        [
            0x9e370000, 0xdaa53c6e, 0x171378dc, 0x5381b54a, 0x8feff1b8, 0xcc5d2e26, 0x08cb6a94, 0x4539a702,
        ],
    ),
    (
        "copy_bf16",
        "n1",
        0x4af55d7f9bf2c2e7,
        [
            0xa5a50000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
    ),
    (
        "copy_bf16",
        "n33",
        0xc17cbaa94965cf03,
        [
            0x9e370000, 0xdaa53c6e, 0x171378dc, 0x5381b54a, 0x8feff1b8, 0xcc5d2e26, 0x08cb6a94, 0x4539a702,
        ],
    ),
    (
        "copy_f16",
        "n65537",
        0x55c8a7c2aa06d882,
        [
            0x9e370000, 0xdaa53c6e, 0x171378dc, 0x5381b54a, 0x8feff1b8, 0xcc5d2e26, 0x08cb6a94, 0x4539a702,
        ],
    ),
    (
        "copy_f16",
        "n1",
        0x4af55d7f9bf2c2e7,
        [
            0xa5a50000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
    ),
    (
        "copy_f16",
        "n33",
        0xc17cbaa94965cf03,
        [
            0x9e370000, 0xdaa53c6e, 0x171378dc, 0x5381b54a, 0x8feff1b8, 0xcc5d2e26, 0x08cb6a94, 0x4539a702,
        ],
    ),
    (
        "cast_bf16_to_f32",
        "n65537",
        0x4c6c388f5011cacc,
        [
            0x00000000, 0x9e370000, 0x3c6e0000, 0xdaa50000, 0x78dc0000, 0x17130000, 0xb54a0000, 0x53810000,
        ],
    ),
    (
        "cast_bf16_to_f32",
        "n1",
        0x4d25767f9dce13f5,
        [
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
    ),
    (
        "cast_bf16_to_f32",
        "n33",
        0x17e9c686b9d99789,
        [
            0x00000000, 0x9e370000, 0x3c6e0000, 0xdaa50000, 0x78dc0000, 0x17130000, 0xb54a0000, 0x53810000,
        ],
    ),
    (
        "cast_f16_to_f32",
        "n65537",
        0xb11a89f212b6037c,
        [
            0x00000000, 0xbbc6e000, 0x3f8dc000, 0xc354a000, 0x471b8000, 0x3ae26000, 0xbea94000, 0x42702000,
        ],
    ),
    (
        "cast_f16_to_f32",
        "n1",
        0x4d25767f9dce13f5,
        [
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
    ),
    (
        "cast_f16_to_f32",
        "n33",
        0xe63e6cccdced459e,
        [
            0x00000000, 0xbbc6e000, 0x3f8dc000, 0xc354a000, 0x471b8000, 0x3ae26000, 0xbea94000, 0x42702000,
        ],
    ),
];

#[test]
fn q4_and_util_kernels_are_bit_identical_to_the_recorded_pre_template_output() {
    with_gpu(|rt| {
        let got = all_rows(rt);
        // Printed in the table's own syntax so a deliberate re-record is a paste.
        for (k, c, h, s) in &got {
            let words: Vec<String> = s.iter().map(|w| format!("{w:#010x}")).collect();
            eprintln!("    (\"{k}\", \"{c}\", {h:#018x}, [{}]),", words.join(", "));
        }
        let mut moved = Vec::new();
        for (g, w) in got.iter().zip(GOLDEN) {
            if (g.0.as_str(), g.1.as_str()) != (w.0, w.1) {
                continue;
            }
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
                    "{} {}: hash {:#018x} != {:#018x}; {detail}",
                    g.0, g.1, g.2, w.2
                ));
            }
        }
        assert!(
            moved.is_empty(),
            "Q4 / util kernel output bits moved:\n{}",
            moved.join("\n")
        );
        let keys: Vec<_> = got.iter().map(|g| (g.0.as_str(), g.1.as_str())).collect();
        let want: Vec<_> = GOLDEN.iter().map(|g| (g.0, g.1)).collect();
        assert_eq!(keys, want, "case list and golden table disagree");
    });
}
