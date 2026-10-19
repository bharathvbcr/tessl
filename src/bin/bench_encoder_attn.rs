//! EmbeddingGemma 2 / BERT / vision-tower encoder attention, kernel against
//! kernel: `encoder_attn`'s rows kernel, its production matrix-unit kernel
//! (f32 operands), its bf16-operand kernel (and the form that stores bf16,
//! `tiled_bf16o`, checked against it before it is timed), and the swept geometries of
//! `kernels/tune/encoder_attn_tune.metal`, on the shapes the models run.
//! (`bench_flash_attn` times the decoder kernels, whose arguments are
//! positions and a causal rule; this takes `lens` and a symmetric window, and
//! names kernels the library's entry points do not expose, hence its own
//! binary.)
//!
//! ```text
//! TESSL_GEMM_TUNE=1 cargo run --release --bin bench_encoder_attn [-- --out PATH]
//! ```
//!
//! Method: every arm of a shape is timed interleaved (`common::run_paired`:
//! forward order on even rounds, reversed on odd), `BENCH_ROUNDS` rounds
//! (default 8) of `BENCH_ITERS` samples (default 5); a sample is `BENCH_REPS`
//! (default 4) back-to-back dispatches queued asynchronously and one
//! `synchronize`, divided by the reps. An arm's figure is the minimum over
//! rounds of the round's median sample (min-of-N, which discards the rounds
//! another process's GPU work landed in); the ratio against the shape's first
//! arm is the median over rounds of the per-round ratios, with the lowest and
//! highest round beside it. A ratio whose range straddles 1.0 is not a result.
//! The baseline's own spread across rounds is printed (`self`) and flagged
//! when it exceeds 10 %.
//!
//! Environment: `BENCH_BASE` (the arm every ratio is against; default the
//! shape's first), `BENCH_SHAPES` (comma list of shape names; default all),
//! `BENCH_ONLY` (comma list of substrings; an arm runs when its name holds
//! one), `BENCH_SWEEP=1` (add the tune geometries; needs a `TESSL_GEMM_TUNE`
//! build), `BENCH_ROUNDS`, `BENCH_ITERS`, `BENCH_REPS`, `BENCH_WARMUP`.
//!
//! FLOP accounting is exact over unmasked (query, key) pairs: `4 * D` per pair
//! per head per sequence (2 D for q.k, 2 D for p.v), the window's two sides
//! and every sequence at its full length (no padding in this bench).

mod common;

use std::sync::Arc;

use common::{
    env_usize, fill_dist, ratio_stats, require_kernels, run_paired, self_spread, Arm, Dist, SELF_SPREAD_LIMIT,
};
use tessl::dispatch::{dispatch_2d_tg, set_f32, set_gpu_buf, set_u32};
use tessl::embedgemma2::{encoder_attn_with, EncoderAttnDims, EncoderAttnKernel};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

/// One model shape the encoder attention runs.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    b: u32,
    t: u32,
    h: u32,
    hkv: u32,
    d: u32,
    window: u32,
}

/// EmbeddingGemma 2 text (4 query heads; sliding layers 2 KV heads of 256 with
/// window 512, global layers 1 KV head of 512) and the vision tower (12 heads
/// of 64, up to 2520 patches, every key).
const SHAPES: [Shape; 7] = [
    Shape {
        name: "text_swa_8x1024",
        b: 8,
        t: 1024,
        h: 4,
        hkv: 2,
        d: 256,
        window: 512,
    },
    Shape {
        name: "text_swa_2x4096",
        b: 2,
        t: 4096,
        h: 4,
        hkv: 2,
        d: 256,
        window: 512,
    },
    Shape {
        name: "text_glob_8x1024",
        b: 8,
        t: 1024,
        h: 4,
        hkv: 1,
        d: 512,
        window: 0,
    },
    Shape {
        name: "text_glob_2x4096",
        b: 2,
        t: 4096,
        h: 4,
        hkv: 1,
        d: 512,
        window: 0,
    },
    Shape {
        name: "vision_1x2520",
        b: 1,
        t: 2520,
        h: 12,
        hkv: 12,
        d: 64,
        window: 0,
    },
    Shape {
        name: "vision_4x2520",
        b: 4,
        t: 2520,
        h: 12,
        hkv: 12,
        d: 64,
        window: 0,
    },
    Shape {
        name: "bert32_8x512",
        b: 8,
        t: 512,
        h: 12,
        hkv: 12,
        d: 32,
        window: 0,
    },
];

/// `kernels/tune/encoder_attn_tune.metal`'s instantiations:
/// (kind, head dim, BQ, BK, NSG) with kind `""` (f32 operands), `"_bf16"` (bf16
/// operands, bf16 P) or `"_bf16p32"` (bf16 operands, f32 P).
const TUNE: &[(&str, u32, u32, u32, u32)] = &[
    ("", 64, 32, 32, 1),
    ("", 64, 32, 32, 2),
    ("", 64, 32, 32, 4),
    ("", 64, 32, 32, 8),
    ("", 64, 32, 64, 1),
    ("", 64, 32, 64, 2),
    ("", 64, 32, 64, 4),
    ("", 64, 32, 64, 8),
    ("", 64, 32, 128, 2),
    ("", 64, 32, 128, 4),
    ("", 64, 32, 128, 8),
    ("", 64, 64, 32, 2),
    ("", 64, 64, 32, 4),
    ("", 64, 64, 32, 8),
    ("", 64, 64, 64, 2),
    ("", 64, 64, 64, 4),
    ("", 64, 64, 64, 8),
    ("", 64, 128, 32, 4),
    ("", 64, 128, 32, 8),
    ("_bf16", 64, 32, 32, 4),
    ("_bf16", 64, 64, 32, 4),
    ("_bf16", 64, 64, 64, 4),
    ("_bf16", 64, 64, 64, 8),
    ("_bf16", 64, 128, 32, 8),
    ("_bf16p32", 64, 32, 32, 4),
    ("_bf16p32", 64, 64, 32, 4),
    ("_bf16p32", 64, 64, 64, 4),
    ("_bf16", 256, 32, 32, 4),
    ("_bf16", 256, 32, 64, 4),
    ("_bf16", 256, 64, 32, 8),
    ("_bf16", 256, 64, 64, 8),
    ("_bf16", 512, 32, 32, 8),
    ("_bf16", 512, 32, 64, 8),
    ("_bf16", 512, 64, 32, 8),
    ("_bf16p32", 256, 32, 32, 4),
    ("_bf16p32", 512, 32, 32, 8),
    ("", 32, 32, 32, 4),
    ("", 32, 64, 32, 4),
    ("", 32, 64, 64, 4),
    ("", 32, 64, 64, 8),
    ("", 32, 128, 32, 8),
    ("_bf16", 64, 32, 64, 8),
    ("_bf16", 64, 32, 128, 4),
    ("_bf16", 64, 32, 128, 8),
    ("_bf16", 64, 64, 32, 8),
    ("_bf16", 64, 64, 64, 2),
    ("_bf16", 64, 64, 64, 16),
    ("_bf16", 64, 32, 32, 2),
    ("_bf16", 256, 32, 128, 4),
    ("_bf16", 256, 32, 64, 2),
    ("_bf16", 256, 32, 64, 8),
    ("_bf16", 256, 64, 64, 4),
    ("_bf16", 256, 16, 64, 4),
    ("_bf16", 256, 16, 64, 2),
    ("_bf16", 512, 32, 128, 8),
    ("_bf16", 512, 32, 64, 16),
    ("_bf16", 512, 32, 128, 16),
    ("_bf16", 512, 16, 64, 8),
    ("_bf16", 512, 16, 128, 8),
    ("_bf16", 512, 64, 64, 16),
    ("_bf16", 64, 32, 32, 1),
    ("_bf16", 64, 32, 64, 1),
    ("_bf16", 64, 32, 64, 2),
    ("_bf16", 64, 32, 128, 2),
    ("_bf16", 64, 16, 32, 1),
    ("_bf16", 64, 16, 32, 2),
    ("_bf16", 64, 16, 64, 1),
    ("_bf16", 64, 16, 64, 2),
    ("_bf16", 64, 64, 32, 2),
    ("_bf16", 256, 32, 32, 2),
    ("_bf16", 256, 16, 32, 2),
    ("_bf16", 256, 16, 32, 4),
    ("_bf16", 512, 16, 32, 4),
    ("_bf16", 512, 16, 64, 4),
    ("", 64, 16, 32, 1),
    ("", 64, 16, 32, 2),
    ("", 64, 16, 64, 2),
    ("", 32, 32, 32, 2),
    ("", 32, 32, 64, 2),
    ("", 32, 32, 32, 1),
    ("", 32, 16, 32, 1),
];

fn tune_name(kind: &str, d: u32, bq: u32, bk: u32, sg: u32) -> String {
    format!("encoder_attn_tune{kind}_h{d}_q{bq}_k{bk}_sg{sg}")
}

impl Shape {
    /// Unmasked (query, key) pairs of one sequence at its full length.
    fn live_pairs(&self) -> u64 {
        let (t, w) = (u64::from(self.t), u64::from(self.window));
        if w == 0 {
            return t * t;
        }
        (0..t).map(|q| (q + w).min(t - 1) - q.saturating_sub(w) + 1).sum()
    }

    fn flop(&self) -> f64 {
        4.0 * f64::from(self.d) * f64::from(self.b * self.h) * self.live_pairs() as f64
    }

    fn dims(&self) -> EncoderAttnDims {
        EncoderAttnDims {
            batch: self.b,
            seq: self.t,
            heads: self.h,
            heads_kv: self.hkv,
            head_dim: self.d,
            window: self.window,
            scale: 1.0,
        }
    }
}

/// bf16 bits of `x`, round to nearest even (finite inputs).
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    ((b.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn f32_buffer(rt: &Arc<GpuRuntime>, data: &[f32]) -> Result<GpuBuffer, String> {
    let b = rt.alloc_buffer(data.len() * 4)?;
    b.write_f32(data);
    Ok(b)
}

fn bf16_buffer(rt: &Arc<GpuRuntime>, data: &[f32]) -> Result<GpuBuffer, String> {
    let b = rt.alloc_buffer(data.len() * 2)?;
    let bits: Vec<u16> = data.iter().map(|&x| bf16_bits(x)).collect();
    b.write_bf16_bits(&bits);
    Ok(b)
}

/// One shape's operands in both precisions.
struct Operands {
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    qb: GpuBuffer,
    kb: GpuBuffer,
    vb: GpuBuffer,
    o: GpuBuffer,
    /// bf16 output, for `EncoderAttnKernel::TiledBf16Out`.
    ob: GpuBuffer,
    lens: GpuBuffer,
}

fn operands(rt: &Arc<GpuRuntime>, s: &Shape) -> Result<Operands, String> {
    let q_n = (s.b * s.t * s.h * s.d) as usize;
    let kv_n = (s.b * s.t * s.hkv * s.d) as usize;
    let q = fill_dist(q_n, 0xA1, Dist::Uniform)
        .iter()
        .map(|x| x * 0.25)
        .collect::<Vec<_>>();
    let k = fill_dist(kv_n, 0xA2, Dist::Uniform)
        .iter()
        .map(|x| x * 0.25)
        .collect::<Vec<_>>();
    let v = fill_dist(kv_n, 0xA3, Dist::Uniform);
    let lens = rt.alloc_buffer(s.b as usize * 4)?;
    lens.write_u32(&vec![s.t; s.b as usize]);
    Ok(Operands {
        q: f32_buffer(rt, &q)?,
        k: f32_buffer(rt, &k)?,
        v: f32_buffer(rt, &v)?,
        qb: bf16_buffer(rt, &q)?,
        kb: bf16_buffer(rt, &k)?,
        vb: bf16_buffer(rt, &v)?,
        o: rt.alloc_buffer(q_n * 4)?,
        ob: rt.alloc_buffer(q_n * 2)?,
        lens,
    })
}

/// Dispatch the entry `name` (BQ and NSG as its geometry spells them) over the
/// encoder attention's buffer slots, f32 or bf16 operands.
fn dispatch_named(
    rt: &Arc<GpuRuntime>,
    name: &str,
    bq: u32,
    nsg: u32,
    s: &Shape,
    bf16: bool,
    ops: &Operands,
) -> Result<(), String> {
    let p = rt.pipeline(name)?;
    let (q, k, v) = if bf16 {
        (&ops.qb, &ops.kb, &ops.vb)
    } else {
        (&ops.q, &ops.k, &ops.v)
    };
    dispatch_2d_tg(
        rt,
        &p,
        s.t.div_ceil(bq) as usize,
        (s.b * s.h) as usize,
        (nsg * 32) as usize,
        |bnd| {
            set_gpu_buf(bnd, q, 0);
            set_gpu_buf(bnd, k, 1);
            set_gpu_buf(bnd, v, 2);
            set_gpu_buf(bnd, &ops.o, 3);
            set_u32(bnd, s.t, 4);
            set_gpu_buf(bnd, &ops.lens, 5);
            set_u32(bnd, s.h, 6);
            set_u32(bnd, s.hkv, 7);
            set_u32(bnd, s.window, 8);
            set_f32(bnd, 1.0, 9);
            set_u32(bnd, s.t, 10);
        },
    )
}

fn add_arm<'a>(
    arms: &mut Vec<Arm<'a>>,
    only: &[String],
    reps: usize,
    name: String,
    run: Box<dyn FnMut() -> Result<(), String> + 'a>,
) {
    if only.is_empty() || only.iter().any(|o| name.contains(o.as_str())) {
        arms.push(Arm { name, reps, run });
    }
}

fn run() -> Result<(), String> {
    let warmup = env_usize("BENCH_WARMUP", 2, 0)?;
    let iters = env_usize("BENCH_ITERS", 5, 1)?;
    let rounds = env_usize("BENCH_ROUNDS", 8, 2)?;
    let reps = env_usize("BENCH_REPS", 4, 1)?;
    let sweep = std::env::var("BENCH_SWEEP").as_deref() == Ok("1");
    let only: Vec<String> = std::env::var("BENCH_ONLY")
        .map(|s| s.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect())
        .unwrap_or_default();
    let shapes_wanted: Vec<String> = std::env::var("BENCH_SHAPES")
        .map(|s| s.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect())
        .unwrap_or_default();
    // The arm every ratio is against (default: the shape's first).
    let base_name = std::env::var("BENCH_BASE").unwrap_or_default();
    let (out_path, coverage_probe) = parse_args(std::env::args().skip(1))?;

    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(true)?;
    if !rt.has_tensorops() {
        return Err("this device has no TensorOps; the matrix-unit kernels cannot run".into());
    }
    if sweep {
        require_kernels(
            &rt,
            "bench_encoder_attn",
            TUNE.iter()
                .map(|&(kind, d, bq, bk, sg)| tune_name(kind, d, bq, bk, sg))
                .collect::<Vec<_>>()
                .iter()
                .map(String::as_str),
        );
    }
    println!(
        "device: {}; {rounds} rounds x {iters} samples, {reps} dispatches per sample, warmup {warmup}",
        rt.device_name()
    );

    let mut json = Vec::new();
    let mut unstable = Vec::new();
    for s in SHAPES
        .iter()
        .filter(|s| shapes_wanted.is_empty() || shapes_wanted.iter().any(|w| w == s.name))
    {
        let ops = operands(&rt, s)?;
        let mut arms: Vec<Arm<'_>> = Vec::new();
        let (rt_ref, ops_ref) = (&rt, &ops);
        let dims = s.dims();
        add_arm(
            &mut arms,
            &only,
            reps,
            "rows".into(),
            Box::new(move || {
                for _ in 0..reps {
                    encoder_attn_with(
                        rt_ref,
                        &ops_ref.q,
                        &ops_ref.k,
                        &ops_ref.v,
                        &ops_ref.o,
                        &ops_ref.lens,
                        dims,
                        EncoderAttnKernel::Rows,
                    )?;
                }
                rt_ref.synchronize()
            }),
        );
        if matches!(s.d, 256 | 512 | 64 | 32) {
            add_arm(
                &mut arms,
                &only,
                reps,
                "tiled".into(),
                Box::new(move || {
                    for _ in 0..reps {
                        encoder_attn_with(
                            rt_ref,
                            &ops_ref.q,
                            &ops_ref.k,
                            &ops_ref.v,
                            &ops_ref.o,
                            &ops_ref.lens,
                            dims,
                            EncoderAttnKernel::Tiled,
                        )?;
                    }
                    rt_ref.synchronize()
                }),
            );
        }
        if matches!(s.d, 256 | 512 | 64) {
            add_arm(
                &mut arms,
                &only,
                reps,
                "tiled_bf16".into(),
                Box::new(move || {
                    for _ in 0..reps {
                        encoder_attn_with(
                            rt_ref,
                            &ops_ref.qb,
                            &ops_ref.kb,
                            &ops_ref.vb,
                            &ops_ref.o,
                            &ops_ref.lens,
                            dims,
                            EncoderAttnKernel::TiledBf16,
                        )?;
                    }
                    rt_ref.synchronize()
                }),
            );
        }
        if matches!(s.d, 256 | 512 | 64) {
            // `bf16o` stores what `tiled_bf16` computes, rounded to bf16 (the
            // library documents the two as the same values up to that cast):
            // check it once before it is timed.
            if only.is_empty() || only.iter().any(|o| "tiled_bf16o".contains(o.as_str())) {
                let q_n = (s.b * s.t * s.h * s.d) as usize;
                ops.o.write_f32(&vec![f32::NAN; q_n]);
                encoder_attn_with(
                    &rt,
                    &ops.qb,
                    &ops.kb,
                    &ops.vb,
                    &ops.o,
                    &ops.lens,
                    dims,
                    EncoderAttnKernel::TiledBf16,
                )?;
                encoder_attn_with(
                    &rt,
                    &ops.qb,
                    &ops.kb,
                    &ops.vb,
                    &ops.ob,
                    &ops.lens,
                    dims,
                    EncoderAttnKernel::TiledBf16Out,
                )?;
                rt.synchronize()?;
                let want: Vec<f64> = ops.o.read_f32().iter().map(|&x| f64::from(x)).collect();
                let got = common::read_bf16(&rt, &ops.ob, want.len())?;
                let what = format!("{} tiled_bf16o", s.name);
                common::check_lane_output(&what, &got)?;
                // One bf16 rounding of the f32 result: 2^-8 relative.
                common::check_against(&what, &got, &want, 1e-6, 8e-3)?;
            }
            add_arm(
                &mut arms,
                &only,
                reps,
                "tiled_bf16o".into(),
                Box::new(move || {
                    for _ in 0..reps {
                        encoder_attn_with(
                            rt_ref,
                            &ops_ref.qb,
                            &ops_ref.kb,
                            &ops_ref.vb,
                            &ops_ref.ob,
                            &ops_ref.lens,
                            dims,
                            EncoderAttnKernel::TiledBf16Out,
                        )?;
                    }
                    rt_ref.synchronize()
                }),
            );
        }
        if sweep {
            // A geometry that is fast because it is wrong is not a result:
            // each candidate's output must match the rows kernel's before it
            // is timed (f32 operands to 1e-4 of the largest value, bf16
            // operands to the rounding of three bf16 inputs, 5e-2).
            encoder_attn_with(
                &rt,
                &ops.q,
                &ops.k,
                &ops.v,
                &ops.o,
                &ops.lens,
                dims,
                EncoderAttnKernel::Rows,
            )?;
            rt.synchronize()?;
            let want = ops.o.read_f32();
            let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-30);
            for &(kind, d, bq, bk, sg) in TUNE.iter().filter(|t| t.1 == s.d) {
                let name = tune_name(kind, d, bq, bk, sg);
                let bf16 = !kind.is_empty();
                ops.o.write_f32(&vec![f32::NAN; want.len()]);
                dispatch_named(&rt, &name, bq, sg, s, bf16, &ops)?;
                rt.synchronize()?;
                let got = ops.o.read_f32();
                let worst = got.iter().zip(&want).fold(0.0f32, |m, (g, w)| {
                    let e = (g - w).abs();
                    if e.is_nan() {
                        f32::INFINITY
                    } else {
                        m.max(e)
                    }
                });
                let tol = if bf16 { 5e-2 } else { 1e-4 };
                if worst > tol * scale {
                    println!(
                        "  REJECTED {name}: max |diff| {worst:.3e} against {scale:.3e} (tolerance {tol:.0e} of it)"
                    );
                    continue;
                }
                let label = name.trim_start_matches("encoder_attn_tune").to_string();
                let shape = *s;
                add_arm(
                    &mut arms,
                    &only,
                    reps,
                    label,
                    Box::new(move || {
                        for _ in 0..reps {
                            dispatch_named(rt_ref, &name, bq, sg, &shape, bf16, ops_ref)?;
                        }
                        rt_ref.synchronize()
                    }),
                );
            }
        }
        if arms.is_empty() {
            continue;
        }
        let result = run_paired(&mut arms, warmup, iters, rounds)?;
        println!(
            "\n{}  b={} t={} h={} hkv={} d={} window={}  ({:.1} GFLOP per dispatch)",
            s.name,
            s.b,
            s.t,
            s.h,
            s.hkv,
            s.d,
            s.window,
            s.flop() / 1e9
        );
        println!(
            "  {:<34} {:>9} {:>9} {:>8}  {:>6}  vs base arm (median of per-round base/arm [lo, hi])",
            "arm", "min ms", "med ms", "TFLOP/s", "self%"
        );
        let base_idx = result.names.iter().position(|n| *n == base_name).unwrap_or(0);
        let base = &result.rounds[base_idx];
        let base_spread = self_spread(base);
        if base_spread > SELF_SPREAD_LIMIT {
            unstable.push(s.name.to_string());
        }
        for (i, name) in result.names.iter().enumerate() {
            let r = &result.rounds[i];
            let min = r.iter().copied().fold(f64::INFINITY, f64::min);
            let med = common::median(r.clone())?;
            let (ratio, lo, hi) = ratio_stats(base, r)?;
            println!(
                "  {:<34} {:>9.3} {:>9.3} {:>8.2}  {:>5.1}%  {:.3}x [{:.3}, {:.3}]",
                name,
                min,
                med,
                s.flop() / (min * 1e-3) / 1e12,
                100.0 * self_spread(r),
                ratio,
                lo,
                hi
            );
            json.push(format!(
                "{{\"shape\":\"{}\",\"arm\":\"{}\",\"min_ms\":{:.6},\"median_ms\":{:.6},\"tflops\":{:.4},\"ratio_vs_base\":{:.5},\"ratio_lo\":{:.5},\"ratio_hi\":{:.5},\"rounds\":[{}]}}",
                s.name,
                name,
                min,
                med,
                s.flop() / (min * 1e-3) / 1e12,
                ratio,
                lo,
                hi,
                r.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",")
            ));
        }
        println!(
            "  base arm ({}) moved {:.1}% against itself across rounds",
            result.names[base_idx],
            100.0 * base_spread
        );
    }
    if let Some(p) = out_path {
        common::atomic_write(std::path::Path::new(&p), format!("[{}]\n", json.join(",\n")).as_bytes())?;
    }
    // A coverage probe (`kernel_coverage` passes `--coverage-probe`) runs two
    // short rounds and is not a measurement: its spread is not a verdict.
    // TESSL_KERNEL_TRACE alone is a user variable and never skips the guard.
    if coverage_probe {
        println!("spread guard skipped: coverage probe");
    } else {
        common::exit_if_unstable("bench_encoder_attn", &unstable);
    }
    Ok(())
}

/// `[--out PATH] [--coverage-probe]`: the JSON path and whether this run is a
/// `kernel_coverage` probe (the only thing that skips the spread guard).
fn parse_args(args: impl Iterator<Item = String>) -> Result<(Option<String>, bool), String> {
    let (mut out_path, mut probe) = (None, false);
    let mut args = args;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_path = Some(args.next().ok_or("--out needs a path")?),
            "--coverage-probe" => probe = true,
            other => {
                return Err(format!(
                    "unknown argument {other:?}; expected --out PATH or --coverage-probe"
                ))
            }
        }
    }
    Ok((out_path, probe))
}

fn main() -> Result<(), String> {
    let outcome = run();
    common::emit_kernel_trace();
    outcome
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    fn parse(a: &[&str]) -> Result<(Option<String>, bool), String> {
        parse_args(a.iter().map(|s| s.to_string()))
    }

    #[test]
    fn only_the_flag_marks_a_coverage_probe_never_the_environment() {
        // TESSL_KERNEL_TRACE is not consulted by the parser at all: with it set
        // to anything, a plain run is still not a probe.
        std::env::set_var("TESSL_KERNEL_TRACE", "1");
        assert_eq!(parse(&[]).unwrap(), (None, false));
        assert_eq!(parse(&["--out", "x.json"]).unwrap(), (Some("x.json".into()), false));
        assert_eq!(parse(&["--coverage-probe"]).unwrap(), (None, true));
        assert!(parse(&["--out"]).is_err());
        assert!(parse(&["--probe"]).is_err());
    }
}
