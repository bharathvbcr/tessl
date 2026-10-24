//! The fused Qwen3.5 decode kernels against the separate ones they replace, on
//! the device: `qwen35::add_rms_norm` (a residual add and the norm after it),
//! `qwen35::gate_up_silu` (the dense gate and up GEMVs and `silu(gate) * up`)
//! and `nn::gemv_q4_mlx_gate_up_silu` (the Q4 form), then a whole decode step
//! through `Decode::set_fusion`.
//!
//! What is bit-exact and what is not, and why:
//!
//! * `add_rms_norm` is bit-exact against `residual_add` then `rms_norm`: the
//!   add is the same f32 add and the sum of squares is reduced in the same
//!   order, so every bit of `resid` and `out` matches.
//! * The Q4 gate/up kernel is bit-exact against two `gemv_q4_mlx_simd`
//!   dispatches and `swiglu`: each projection runs that kernel's accumulation.
//! * The dense gate/up kernel is not: it sums k in its own fixed order (32
//!   k-lanes, folded by shuffles), not the TensorOps GEMM's. Both are f32 sums
//!   of the same products, so they agree to f32 rounding of the two sums.
//!   Against an f64 oracle each stays inside the worst-case dot-product bound
//!   (`gamma_k = k * 2^-24` of the sum of absolute products); the kernels agree
//!   with each other to far less, which the test prints and bounds at `1e-5` of
//!   the vector's peak (f32 output) or one bf16 rounding step (bf16 output,
//!   where a sum near a rounding boundary can round the other way).
//! * A whole step: `add_norm` alone is bit-identical (logits compared as bits);
//!   `gate_up` on dense weights moves the logits by that f32 rounding, bounded
//!   as `tests/qwen35_model.rs` bounds a decode against a forward (f32 `1e-5`,
//!   bf16 `2e-2`), with the same top-1 token wherever the top two logits are
//!   further apart than the bound. On Q4 the whole step is bit-identical.
//! * Dispatches per token are counted with `infer_trace`, the way
//!   `bench_qwen35_layers --decode` counts them.

mod common;

use common::{buf, buf_bf16, empty, random_f32, round_trip_bf16, tensor_bf16, tensor_f32, with_gpu};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tessl::gemm::{gemm, GemmBackend};
use tessl::infer_trace;
use tessl::nn::{self, Q4MlxBank, Q4MlxLayout, QuantShape};
use tessl::qwen35::{self, AttnProjLayout, Cols, GdnProjLayout, OutCols};
use tessl::qwen35_model::{DecodeFusion, LayerKind, Precision, Qwen35Config, Qwen35Model};
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16, DType, GpuBuffer};
use tessl::GpuRuntime;

const EPS: f32 = 1e-6;

fn bf16_vec(b: &GpuBuffer, n: usize) -> Vec<f32> {
    b.try_contents_u16().unwrap()[..n]
        .iter()
        .map(|&h| bf16_bits_to_f32(h))
        .collect()
}

fn f32_vec(b: &GpuBuffer, n: usize) -> Vec<f32> {
    b.try_contents_f32().unwrap()[..n].to_vec()
}

fn out_buf(rt: &Arc<GpuRuntime>, dtype: DType, n: usize) -> GpuBuffer {
    match dtype {
        DType::BF16 => {
            let b = rt.alloc_buffer(n.max(1) * 2).unwrap();
            b.zero();
            b
        }
        _ => empty(rt, n),
    }
}

fn read_out(rt: &Arc<GpuRuntime>, b: &GpuBuffer, dtype: DType, n: usize) -> Vec<f32> {
    rt.synchronize().unwrap();
    match dtype {
        DType::BF16 => bf16_vec(b, n),
        _ => f32_vec(b, n),
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ------------------------------------------------------------ add + norm ---

/// `add_rms_norm` against `residual_add` then `rms_norm`, every bit of `resid`
/// and `out`, at f32 and bf16 `out`, rows and dims that leave lanes idle (1, 7,
/// 100), a dim above one threadgroup's walk (2048), and a residual whose
/// magnitudes span decades.
#[test]
fn add_rms_norm_is_bit_identical_to_the_add_then_the_norm() {
    with_gpu(|rt| {
        for dtype in [DType::F32, DType::BF16] {
            for (case, &(rows, dim)) in [(1usize, 2048usize), (3, 100), (2, 7), (1, 1), (4, 64), (2, 4100)]
                .iter()
                .enumerate()
            {
                let n = rows * dim;
                let seed = 0x5100 + case as u64;
                let y = random_f32(n, seed);
                let resid: Vec<f32> = random_f32(n, seed + 1)
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * [0.01f32, 1.0, 30.0, 1.0e3][i % 4])
                    .collect();
                let w = random_f32(dim, seed + 2);
                let (yb, wb) = (buf(rt, &y), buf(rt, &w));

                let (r_ref, out_ref) = (buf(rt, &resid), out_buf(rt, dtype, n));
                qwen35::residual_add(
                    rt,
                    Cols::dense(&yb, dim as u32),
                    Cols::dense(&r_ref, dim as u32),
                    rows as u32,
                    dim as u32,
                )
                .unwrap();
                qwen35::rms_norm(rt, &r_ref, &wb, &out_ref, dtype, rows as u32, dim as u32, EPS).unwrap();

                let (r_fused, out_fused) = (buf(rt, &resid), out_buf(rt, dtype, n));
                rt.take_dispatch_count();
                qwen35::add_rms_norm(rt, &yb, &r_fused, &wb, &out_fused, dtype, rows as u32, dim as u32, EPS).unwrap();
                assert_eq!(rt.take_dispatch_count(), 1, "the fused pass is one dispatch");

                assert_eq!(
                    bits(&read_out(rt, &r_fused, DType::F32, n)),
                    bits(&read_out(rt, &r_ref, DType::F32, n)),
                    "{dtype:?} {rows}x{dim}: resid"
                );
                assert_eq!(
                    bits(&read_out(rt, &out_fused, dtype, n)),
                    bits(&read_out(rt, &out_ref, dtype, n)),
                    "{dtype:?} {rows}x{dim}: out"
                );
            }
        }
    });
}

#[test]
fn add_rms_norm_refuses_aliases_and_short_buffers_before_encoding() {
    with_gpu(|rt| {
        let (y, r, w) = (buf(rt, &[1.0; 8]), buf(rt, &[1.0; 8]), buf(rt, &[0.0; 8]));
        let out = out_buf(rt, DType::F32, 8);
        rt.take_dispatch_count();
        for (what, res) in [
            (
                "out aliases resid",
                qwen35::add_rms_norm(rt, &y, &r, &w, &r, DType::F32, 1, 8, EPS),
            ),
            (
                "out aliases y",
                qwen35::add_rms_norm(rt, &y, &r, &w, &y, DType::F32, 1, 8, EPS),
            ),
            (
                "resid aliases y",
                qwen35::add_rms_norm(rt, &y, &y, &w, &out, DType::F32, 1, 8, EPS),
            ),
            (
                "short y",
                qwen35::add_rms_norm(rt, &y, &r, &w, &out, DType::F32, 2, 8, EPS),
            ),
            (
                "bad eps",
                qwen35::add_rms_norm(rt, &y, &r, &w, &out, DType::F32, 1, 8, f32::NAN),
            ),
            (
                "f16 out",
                qwen35::add_rms_norm(rt, &y, &r, &w, &out, DType::F16, 1, 8, EPS),
            ),
        ] {
            assert!(res.is_err(), "{what} was accepted");
        }
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

// ------------------------------------------------------- dense gate / up ---

/// `(k, n)`: the 2B's own shape; a model-sized pair; `n` that is not a multiple
/// of 8 (scalar tail) and below one threadgroup's 64 columns; `k` that is not a
/// multiple of the kernel's 32 k-lanes; a single column and a single k.
const DENSE_SHAPES: &[(usize, usize)] = &[
    (2048, 6144),
    (64, 128),
    (37, 45),
    (33, 8),
    (100, 200),
    (5, 1),
    (1, 7),
    (256, 130),
];

fn silu64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// One bf16 rounding step either way: adjacent bf16 values are 2^-7 apart at
/// worst, relative.
fn within_one_bf16_step(a: f32, b: f32) -> bool {
    (a - b).abs() <= a.abs().max(b.abs()) * 2f32.powi(-7) + f32::MIN_POSITIVE
}

#[test]
fn dense_gate_up_silu_matches_the_gemm_pair_and_the_f64_oracle() {
    with_gpu(|rt| {
        for dtype in [DType::F32, DType::BF16] {
            let bf16 = dtype == DType::BF16;
            let operand = |v: Vec<f32>| if bf16 { round_trip_bf16(&v) } else { v };
            let tensor = |shape: &[usize], d: &[f32]| {
                if bf16 {
                    tensor_bf16(rt, shape, d)
                } else {
                    tensor_f32(rt, shape, d)
                }
            };
            for (case, &(k, n)) in DENSE_SHAPES.iter().enumerate() {
                let seed = 0x5200 + case as u64 * 8;
                let x = operand(random_f32(k, seed));
                let wg: Vec<f32> = operand(random_f32(k * n, seed + 1).iter().map(|v| v * 0.1).collect());
                let wu: Vec<f32> = operand(random_f32(k * n, seed + 2).iter().map(|v| v * 0.1).collect());
                let (xt, gt, ut) = (tensor(&[1, k], &x), tensor(&[k, n], &wg), tensor(&[k, n], &wu));

                // The separate path, as Qwen35Model::mlp runs it.
                let (mg, mu) = (
                    rt.alloc_tensor_f32(&[1, n]).unwrap(),
                    rt.alloc_tensor_f32(&[1, n]).unwrap(),
                );
                gemm(&xt, &gt, &mg, GemmBackend::TensorOps).unwrap();
                gemm(&xt, &ut, &mu, GemmBackend::TensorOps).unwrap();
                let mid_ref = out_buf(rt, dtype, n);
                qwen35::swiglu(
                    rt,
                    Cols::dense(&mg.buffer, n as u32),
                    Cols::dense(&mu.buffer, n as u32),
                    OutCols {
                        cols: Cols::dense(&mid_ref, n as u32),
                        dtype,
                    },
                    1,
                    n as u32,
                )
                .unwrap();

                let mid = out_buf(rt, dtype, n);
                rt.take_dispatch_count();
                qwen35::gate_up_silu(rt, &xt.buffer, &gt.buffer, &ut.buffer, &mid, dtype, k as u32, n as u32).unwrap();
                assert_eq!(rt.take_dispatch_count(), 1, "the fused front half is one dispatch");

                let (fused, unfused) = (read_out(rt, &mid, dtype, n), read_out(rt, &mid_ref, dtype, n));
                let mut oracle = Vec::with_capacity(n);
                let mut bound = Vec::with_capacity(n);
                for j in 0..n {
                    let (mut g, mut u, mut sg, mut su) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                    for i in 0..k {
                        let (a, b) = (
                            f64::from(x[i]) * f64::from(wg[i * n + j]),
                            f64::from(x[i]) * f64::from(wu[i * n + j]),
                        );
                        g += a;
                        u += b;
                        sg += a.abs();
                        su += b.abs();
                    }
                    oracle.push(silu64(g) * u);
                    // gamma_k bound on each sum, through the product's derivative (silu' <= 1.1).
                    let gamma = k as f64 * 2f64.powi(-24);
                    let sum_err = 1.1 * u.abs() * gamma * sg + silu64(g).abs() * gamma * su;
                    let out_round = oracle[j].abs() * if bf16 { 2f64.powi(-8) } else { 2f64.powi(-24) };
                    bound.push(1.2 * (sum_err + out_round) + 1e-30);
                }
                let peak = oracle.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let label = format!("{dtype:?} k={k} n={n}");
                let (mut worst_pair, mut flips) = (0.0f64, 0usize);
                for j in 0..n {
                    for (name, v) in [("fused", fused[j]), ("unfused", unfused[j])] {
                        let e = (f64::from(v) - oracle[j]).abs();
                        assert!(
                            e <= bound[j],
                            "{label} [{j}] {name}: err {e:.3e} over the gamma_k bound {:.3e}",
                            bound[j]
                        );
                    }
                    let d = f64::from((fused[j] - unfused[j]).abs());
                    worst_pair = worst_pair.max(d / peak.max(f64::MIN_POSITIVE));
                    if bf16 {
                        assert!(
                            within_one_bf16_step(fused[j], unfused[j]),
                            "{label} [{j}]: {} vs {}",
                            fused[j],
                            unfused[j]
                        );
                        flips += usize::from(fused[j] != unfused[j]);
                    } else {
                        assert!(
                            d <= 1e-5 * peak,
                            "{label} [{j}]: fused {} vs unfused {} (peak {peak:.3e})",
                            fused[j],
                            unfused[j]
                        );
                    }
                }
                eprintln!(
                    "{label}: fused vs gemm pair worst {worst_pair:.2e} of peak, {flips} bf16 rounding flips of {n}"
                );
            }
        }
    });
}

#[test]
fn dense_gate_up_silu_refuses_aliases_and_short_buffers_before_encoding() {
    with_gpu(|rt| {
        let (x, w, o) = (buf(rt, &[1.0; 4]), buf(rt, &[0.5; 16]), out_buf(rt, DType::F32, 4));
        rt.take_dispatch_count();
        for (what, res) in [
            (
                "out aliases x",
                qwen35::gate_up_silu(rt, &x, &w, &w, &x, DType::F32, 4, 4),
            ),
            (
                "out aliases a weight",
                qwen35::gate_up_silu(rt, &x, &w, &o, &o, DType::F32, 4, 4),
            ),
            (
                "short weight",
                qwen35::gate_up_silu(rt, &x, &w, &w, &o, DType::F32, 4, 5),
            ),
            ("short x", qwen35::gate_up_silu(rt, &x, &w, &w, &o, DType::F32, 5, 3)),
            ("f16", qwen35::gate_up_silu(rt, &x, &w, &w, &o, DType::F16, 4, 4)),
        ] {
            assert!(res.is_err(), "{what} was accepted");
        }
        assert_eq!(rt.take_dispatch_count(), 0);
        qwen35::gate_up_silu(rt, &x, &w, &w, &o, DType::F32, 0, 4).unwrap();
        assert_eq!(rt.take_dispatch_count(), 0, "k = 0 encodes nothing");
    });
}

// ----------------------------------------------------------- Q4 gate / up ---

fn rand_bank(rt: &Arc<GpuRuntime>, rows: usize, cols: usize, group: usize, seed: u64) -> (GpuBuffer, GpuBuffer) {
    let r = random_f32(rows * cols / 2 + rows * (cols / group) * 2, seed);
    let packed: Vec<u8> = r[..rows * cols / 2]
        .iter()
        .map(|v| ((v * 127.0) as i32 & 0xff) as u8)
        .collect();
    let sb: Vec<f32> = r[rows * cols / 2..]
        .chunks(2)
        .flat_map(|p| [0.002 + 0.05 * p[0].abs(), 0.5 * p[1]])
        .collect();
    let p = rt.alloc_buffer(packed.len().max(4)).unwrap();
    p.write_bytes(&packed);
    let s = rt.alloc_buffer(sb.len().max(1) * 2).unwrap();
    s.write_bf16_bits(&f32_slice_to_bf16(&sb));
    (p, s)
}

/// The Q4 gate/up kernel against two `gemv_q4_mlx_simd` and `swiglu`: every bit,
/// f32 and bf16 mid, odd row counts, a K tail that leaves upper lanes idle.
#[test]
fn q4_gate_up_silu_is_bit_identical_to_two_gemvs_and_swiglu() {
    with_gpu(|rt| {
        for &(rows, cols, group) in &[
            (37usize, 544usize, 32usize),
            (64, 1024, 64),
            (13, 96, 32),
            (6, 512, 128),
            (9, 2112, 64),
        ] {
            let sh = QuantShape {
                rows: rows as u32,
                cols: cols as u32,
                group_size: group as u32,
            };
            let x = buf_bf16(
                rt,
                &random_f32(cols, 0x5300 + cols as u64)
                    .iter()
                    .map(|v| v * 2.0)
                    .collect::<Vec<_>>(),
            );
            let (gp, gs) = rand_bank(rt, rows, cols, group, 0x5400 + rows as u64);
            let (up, us) = rand_bank(rt, rows, cols, group, 0x5500 + rows as u64);
            let (gb, ub) = (
                Q4MlxBank {
                    packed: &gp,
                    scales_biases: &gs,
                },
                Q4MlxBank {
                    packed: &up,
                    scales_biases: &us,
                },
            );
            let (g, u) = (empty(rt, rows), empty(rt, rows));
            nn::gemv_q4_mlx_simd(rt, gb, &x, &g, sh, Q4MlxLayout::RowMajor, None).unwrap();
            nn::gemv_q4_mlx_simd(rt, ub, &x, &u, sh, Q4MlxLayout::RowMajor, None).unwrap();
            for dtype in [DType::F32, DType::BF16] {
                let want = out_buf(rt, dtype, rows);
                qwen35::swiglu(
                    rt,
                    Cols::dense(&g, rows as u32),
                    Cols::dense(&u, rows as u32),
                    OutCols {
                        cols: Cols::dense(&want, rows as u32),
                        dtype,
                    },
                    1,
                    rows as u32,
                )
                .unwrap();
                let got = out_buf(rt, dtype, rows);
                rt.take_dispatch_count();
                nn::gemv_q4_mlx_gate_up_silu(rt, gb, ub, &x, &got, sh, dtype == DType::BF16).unwrap();
                assert_eq!(rt.take_dispatch_count(), 1);
                assert_eq!(
                    bits(&read_out(rt, &got, dtype, rows)),
                    bits(&read_out(rt, &want, dtype, rows)),
                    "{dtype:?} {rows}x{cols} g{group}"
                );
            }
        }
    });
}

// ------------------------------------------------------------ whole step ---

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

/// Four layers (GDN, attention, GDN, attention) at the tiny model's widths and
/// two GDN value heads per key head, with the intermediate size a parameter: 128
/// is a multiple of the kernel's 8-column vectors, 100 is not (scalar path).
fn config(intermediate: u32) -> Qwen35Config {
    let tiny = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    Qwen35Config {
        layers: vec![
            LayerKind::LinearAttention,
            LayerKind::FullAttention,
            LayerKind::LinearAttention,
            LayerKind::FullAttention,
        ],
        gdn: GdnProjLayout::new(1, 2, 128).unwrap(),
        attn: AttnProjLayout::new(2, 1, 256).unwrap(),
        intermediate,
        ..tiny
    }
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
        .0
}

fn rel(got: &[f32], want: &[f32]) -> f64 {
    let peak = want.iter().fold(0.0f64, |m, &w| m.max(f64::from(w).abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((f64::from(g) - f64::from(w)).abs()));
    err / peak.max(f64::MIN_POSITIVE)
}

/// Runs `steps` greedy steps from a prefill of `prompt` ids at `fusion`,
/// feeding each session the token the UNFUSED session chose, so both see the
/// same inputs; returns the logits of every step.
fn run_steps(model: &Qwen35Model, prompt: &[u32], feed: &[u32], fusion: DecodeFusion) -> Vec<Vec<f32>> {
    let (mut dec, first) = model.prefill(prompt, feed.len() as u32).unwrap();
    dec.set_fusion(fusion);
    assert_eq!(dec.fusion(), fusion);
    let mut rows = vec![first];
    for &id in feed {
        rows.push(dec.step(id).unwrap());
    }
    rows
}

/// Greedy tokens of the unfused session, as the feed for every session.
fn greedy_feed(model: &Qwen35Model, prompt: &[u32], steps: usize) -> Vec<u32> {
    let (mut dec, first) = model.prefill(prompt, steps as u32).unwrap();
    dec.set_fusion(DecodeFusion::NONE);
    let mut next = argmax(&first) as u32;
    let mut feed = Vec::new();
    for _ in 0..steps {
        feed.push(next);
        next = argmax(&dec.step(next).unwrap()) as u32;
    }
    feed
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 64).collect()
}

/// A row of fused logits against the unfused one: within `bound` of its peak,
/// and the same top-1 where the unfused row's top two are further apart than
/// twice what the bound allows.
fn check_row(label: &str, got: &[f32], want: &[f32], bound: f64) -> f64 {
    let r = rel(got, want);
    assert!(r <= bound, "{label}: rel err {r:.3e} > {bound:.0e}");
    let (top, mut second) = (argmax(want), f32::NEG_INFINITY);
    for (i, &v) in want.iter().enumerate() {
        if i != top && v > second {
            second = v;
        }
    }
    let peak = want.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
    if f64::from(want[top] - second) > 2.0 * bound * f64::from(peak) {
        assert_eq!(argmax(got), top, "{label}: top-1");
    }
    r
}

#[test]
fn fused_decode_matches_the_unfused_step_on_random_models() {
    const STEPS: usize = 6;
    with_gpu(|rt| {
        for precision in [Precision::F32, Precision::Bf16] {
            let bound = if precision == Precision::F32 { 1e-5 } else { 2e-2 };
            for inter in [128u32, 100] {
                let model = Qwen35Model::random(rt, config(inter), precision, 7 + u64::from(inter)).unwrap();
                let mut worst = [0.0f64; 3];
                for n in [1usize, 5, 70] {
                    let ids = prompt(n);
                    let feed = greedy_feed(&model, &ids, STEPS);
                    let base = run_steps(&model, &ids, &feed, DecodeFusion::NONE);
                    let again = run_steps(&model, &ids, &feed, DecodeFusion::NONE);
                    for (a, b) in base.iter().zip(&again) {
                        assert_eq!(bits(a), bits(b), "the unfused step is not deterministic");
                    }
                    let modes = [
                        (
                            "add_norm",
                            DecodeFusion {
                                add_norm: true,
                                gate_up: false,
                            },
                        ),
                        (
                            "gate_up",
                            DecodeFusion {
                                add_norm: false,
                                gate_up: true,
                            },
                        ),
                        ("all", DecodeFusion::ALL),
                    ];
                    for (m, (name, fusion)) in modes.iter().enumerate() {
                        let got = run_steps(&model, &ids, &feed, *fusion);
                        for (k, (g, w)) in got.iter().zip(&base).enumerate() {
                            let label = format!("{precision:?} inter {inter} prefill {n} + {k} [{name}]");
                            if !fusion.gate_up {
                                // Same arithmetic, so the same bits.
                                assert_eq!(bits(g), bits(w), "{label}: add_norm alone is not bit-identical");
                            } else {
                                worst[m] = worst[m].max(check_row(&label, g, w, bound));
                            }
                        }
                    }
                }
                eprintln!(
                    "{precision:?} inter {inter}: fused vs unfused logits, worst rel err gate_up {:.2e}, all {:.2e} (bound {bound:.0e}); add_norm alone bit-identical",
                    worst[1], worst[2]
                );
            }
        }
    });
}

/// Dispatches per decode step, counted with `infer_trace` over separate traced
/// steps as `bench_qwen35_layers --decode` counts them. The fused step saves
/// two dispatches per layer for the gate/up (the gate GEMV, the up GEMV and the
/// SwiGLU become one), and, in the exact-f32 forward only, the two residual adds
/// per layer that stop being dispatches of their own (a mixer's output before
/// the post norm, and the down projection's before the next input norm or the
/// final one). The bf16 forward folds both adds into the projections.
/// Everything else a step counts (commits, waits, cold allocations) is unchanged.
#[test]
fn fused_decode_saves_exactly_the_dispatches_it_replaces() {
    with_gpu(|rt| {
        for precision in [Precision::F32, Precision::Bf16] {
            let cfg = config(128);
            let layers = cfg.layers.len() as u64;
            let model = Qwen35Model::random(rt, cfg, precision, 5).unwrap();
            let ids = prompt(8);
            let feed = greedy_feed(&model, &ids, 4);
            let count = |fusion: DecodeFusion| {
                // Async encode, as `bench_qwen35_layers --decode` and a real decode
                // loop run: one command buffer, one wait per step. The runtime's
                // default (sync) mode commits and waits after every dispatch that
                // is not already inside a batching scope, so there commits and
                // waits follow the dispatch count (measured: 57 -> 49 on this
                // model's f32 add_norm when 8 plain dispatches went) and say
                // nothing about the fusion.
                rt.set_async_encode(true).unwrap();
                let (mut dec, _) = model.prefill(&ids, 4).unwrap();
                dec.set_fusion(fusion);
                dec.step(feed[0]).unwrap();
                infer_trace::set_enabled(true);
                let before = infer_trace::snapshot();
                dec.step(feed[1]).unwrap();
                let s = infer_trace::snapshot().since(&before);
                infer_trace::set_enabled(false);
                rt.set_async_encode(false).unwrap();
                s
            };
            let none = count(DecodeFusion::NONE);
            let adds = if precision == Precision::F32 { 2 * layers } else { 0 };
            for (name, fusion, saved) in [
                (
                    "add_norm",
                    DecodeFusion {
                        add_norm: true,
                        gate_up: false,
                    },
                    adds,
                ),
                (
                    "gate_up",
                    DecodeFusion {
                        add_norm: false,
                        gate_up: true,
                    },
                    2 * layers,
                ),
                ("all", DecodeFusion::ALL, 2 * layers + adds),
            ] {
                let got = count(fusion);
                assert_eq!(
                    got.dispatches,
                    none.dispatches - saved,
                    "{precision:?} {name}: {} dispatches against the unfused {}",
                    got.dispatches,
                    none.dispatches
                );
                assert_eq!(
                    (got.commits, got.sync_waits, got.cold_allocs),
                    (none.commits, none.sync_waits, none.cold_allocs),
                    "{precision:?} {name}: commits, waits and cold allocations"
                );
                assert_eq!(
                    (got.commits, got.sync_waits, got.cold_allocs),
                    (1, 1, 3),
                    "{precision:?} {name}: an async step is one commit, one wait, three cold allocations"
                );
            }
            eprintln!(
                "{precision:?}: {} dispatches per token unfused, {} fused (4 layers)",
                none.dispatches,
                none.dispatches - 2 * layers - adds
            );
        }
    });
}
