//! End-to-end latency and memory of `tessl::embedgemma2` on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   cargo run --release --bin bench_embedgemma2
//! ```
//!
//! `--kernels` instead runs the text tower's four fused row passes on random
//! operands, with no checkpoint (see `run_text_kernels`).
//!
//! `--attn-kernel=rows|tiled` instead times the encoder attention kernel alone
//! at the checkpoint's layer shapes, with no weights (see `run_attn_kernel`).
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,`. A workload is one
//! `encode` call: terms joined by `+`, each `BxT` (`B` sequences of `T`
//! tokens) or `BxLO-HI` (`B` sequences, the `i`-th of
//! `LO + (i * 2654435761) % (HI - LO + 1)` tokens). The uniform workloads run
//! one forward; the ragged ones exercise `encode`'s split into forwards
//! (`forwards` in the output says how many ran). The default:
//!
//! * uniform: `1x16`, `64x32`, `32x256`, `8x1024`, `2x4096`;
//! * ragged: `128x8-512` (short texts, more padded rows than one forward
//!   holds), `1x4096+63x32` (a long document beside short queries), and
//!   `1x6147+1x1658+6x10-42` (the shape of `tools/embedgemma2_ref`'s eight
//!   texts).
//!
//! `BENCH_ITERS` (default 10) and `BENCH_WARMUP` (default 3) are per
//! workload. `encode` queues its dispatches and waits once whatever the
//! runtime's `async_encode` flag says (it opens a `GpuRuntime::async_scope`),
//! so `BENCH_ASYNC=1` (`set_async_encode(true)`) no longer changes how it
//! runs; it is kept so older result files' lanes can be re-run. Each object
//! carries `"async":0|1`. `BENCH_SERIALIZED=1` (`set_serialized_dispatch`)
//! makes every dispatch commit and wait, which is how `encode` ran on a
//! default runtime before the scope: the baseline for what queueing is worth.
//! Each object carries `"serialized":0|1`. `BENCH_PLE_RUN_BYTES` (default the
//! library's `PLE_RUN_BYTES`) calls `EmbedGemma2Model::set_ple_run_bytes`, to
//! measure the per-layer-input bound's time and memory cost in one build.
//! `BENCH_PRECISION=f32|bf16` (default `f32`) picks the
//! operand precision the model is loaded with
//! ([`EmbedGemma2Model::load_with`]: `GemmOperands::ExactF32` or `Bf16`); the
//! object's `backend` says which (`tessl-f32`, `tessl-bf16`) and `weight_mib`
//! is the model's weight bytes on the device. Prints one JSON array on stdout,
//! one object per workload: `{workload, backend, ms_min, ms_median, iters,
//! async, serialized, forwards, weight_mib, device_peak_mib, peak_footprint_mib}`.
//! `device_peak_mib` is the device's peak allocation
//! during that workload above the loaded model
//! ([`GpuRuntime::peak_allocated_bytes`]); `peak_footprint_mib` is the
//! process's lifetime peak physical footprint so far (what `/usr/bin/time -l`
//! reports as `peak memory footprint`), so only its last value covers the
//! whole run. `bench_paired` alternates this binary with the PyTorch lane
//! (`bench/embedgemma2_torch.py`, the same token ids) or with another build
//! of itself, round by round.
//!
//! What is timed: one `encode` call, from host token ids to host embeddings:
//! activation allocation, upload, the forwards, pooling, projection and
//! normalization, and the read back. Not timed: tokenization (both lanes take
//! ids) and the model load.
//!
//! Before anything is timed, every workload's embeddings must be finite and of
//! unit norm; the run aborts otherwise, so a forward that wrote nothing cannot
//! post the best time.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2::{encoder_attn_with, EmbedGemma2Config, EmbedGemma2Model, EncoderAttnDims, EncoderAttnKernel};
use tessl::gemm::GemmOperands;
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

use common::{env_usize, peak_footprint};

const DEFAULT_WORKLOADS: &str = "1x16,64x32,32x256,8x1024,2x4096,128x8-512,1x4096+63x32,1x6147+1x1658+6x10-42";

const MIB: f64 = (1u64 << 20) as f64;

/// The token ids every lane uses for sequence `seq` of a workload: BOS, then
/// a spread of ordinary ids, then EOS.
fn ids(seq: usize, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| {
            if i == 0 {
                2
            } else if i == len - 1 {
                1
            } else {
                (1000 + (seq * 7919 + i * 104_729) % 200_000) as u32
            }
        })
        .collect()
}

/// The sequence lengths of one workload (see the module docs for the grammar).
fn parse_workload(w: &str) -> Result<Vec<usize>, String> {
    let mut lens = Vec::new();
    for term in w.trim().split('+') {
        let (b, t) = term
            .trim()
            .split_once('x')
            .ok_or_else(|| format!("workload {w:?}: term {term:?} is not BxT or BxLO-HI"))?;
        let num = |s: &str| -> Result<usize, String> { s.parse().map_err(|e| format!("workload {w:?}: {s:?}: {e}")) };
        let b = num(b)?;
        let (lo, hi) = match t.split_once('-') {
            Some((lo, hi)) => (num(lo)?, num(hi)?),
            None => (num(t)?, num(t)?),
        };
        if b == 0 || lo < 2 || hi < lo {
            return Err(format!(
                "workload {w:?}: term {term:?} needs at least 1 sequence and 2 <= LO <= HI tokens"
            ));
        }
        lens.extend((0..b).map(|i| lo + (i * 2_654_435_761) % (hi - lo + 1)));
    }
    Ok(lens)
}

/// Workloads `--attn-kernel` times by default: the `encode` defaults' uniform
/// shapes above the window (512), the long document of `1x4096+63x32`, and
/// `8x64-1024`, whose row lengths run from 64 to 1024 under a padded length of
/// 1024, so padding queries are exercised.
const DEFAULT_ATTN_SHAPES: &str = "64x32,32x256,8x1024,2x4096,1x6147,8x64-1024";

/// The tiled kernel must match the rows kernel on the same inputs to within
/// this fraction of the largest rows output before it is timed. Both are f32
/// kernels that agree with the f64 reference to 2e-5 (`tests/embedgemma2_kernels.rs`
/// `KERNEL_REL`); five times that is the allowance for their difference.
const ATTN_PARITY_REL: f64 = 1e-4;

/// One `BENCH_ATTN_SHAPES` entry: `lens.len()` sequences padded to `t`.
#[derive(Debug, PartialEq)]
struct AttnShape {
    /// The metric-name part: `b{B}_t{T}`, plus `_len{sum of lengths}` when ragged.
    key: String,
    t: usize,
    lens: Vec<usize>,
}

/// All of `BENCH_ATTN_SHAPES` before anything runs: each entry a workload of
/// [`parse_workload`], the padded length its longest row, no two entries the
/// same metric name.
fn parse_attn_shapes(spec: &str) -> Result<Vec<AttnShape>, String> {
    let mut shapes: Vec<AttnShape> = Vec::new();
    for entry in spec.split(',') {
        let lens = parse_workload(entry.trim()).map_err(|e| format!("BENCH_ATTN_SHAPES: {e}"))?;
        let t = lens.iter().copied().max().unwrap_or(0);
        let ragged = lens.iter().any(|&l| l != t);
        let key = if ragged {
            format!("b{}_t{t}_len{}", lens.len(), lens.iter().sum::<usize>())
        } else {
            format!("b{}_t{t}", lens.len())
        };
        if shapes.iter().any(|s| s.key == key) {
            return Err(format!(
                "BENCH_ATTN_SHAPES: {entry:?} repeats an earlier entry (metric {key})"
            ));
        }
        shapes.push(AttnShape { key, t, lens });
    }
    Ok(shapes)
}

/// The output `got` of the tiled kernel against `want` of the rows kernel for
/// `lens.len()` sequences of `t` positions (`[B, T, H*D]`, `row` = `H*D`
/// floats per position). Every element must be finite, a padding position
/// (`pos >= lens[b]`) must be `+0.0` in both, and the largest absolute
/// difference must be within [`ATTN_PARITY_REL`] of the largest `want`.
/// Returns that difference.
fn attn_parity(want: &[f32], got: &[f32], lens: &[usize], t: usize, row: usize) -> Result<f64, String> {
    let n = lens.len() * t * row;
    if want.len() != n || got.len() != n {
        return Err(format!("output lengths {} and {}, expected {n}", want.len(), got.len()));
    }
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        let (b, pos) = (i / (t * row), i / row % t);
        if !w.is_finite() || !g.is_finite() {
            return Err(format!("element {i}: rows {w}, tiled {g} is not finite"));
        }
        if pos >= lens[b] && !(w == 0.0 && w.is_sign_positive() && g == 0.0 && g.is_sign_positive()) {
            return Err(format!(
                "padding position {pos} of sequence {b}: rows {w}, tiled {g}, expected +0"
            ));
        }
        worst = worst.max((f64::from(w) - f64::from(g)).abs());
        scale = scale.max(f64::from(w).abs());
    }
    if scale == 0.0 {
        return Err("the rows output is all zero".into());
    }
    if worst > ATTN_PARITY_REL * scale {
        return Err(format!(
            "tiled differs from rows by {worst:.3e}, over {ATTN_PARITY_REL:.0e} of the largest output {scale:.3e}"
        ));
    }
    Ok(worst)
}

/// `--attn-kernel=rows|tiled`: the encoder attention kernel alone, per
/// distinct layer kind of the checkpoint's `config.json` (no weights are
/// read), at each shape of `BENCH_ATTN_SHAPES` (`shapes`, already parsed).
/// One `METRIC encoder_attn_{sliding|global}_h{D}_{shape key}_ms` per kind
/// per shape, the median of `BENCH_ITERS` buffers of `BENCH_ATTN_REPS`
/// launches; the metric names carry no kernel, so `bench_paired --format
/// metrics` lanes `rows` and `tiled` (or two builds) compare name for name.
/// Before timing, the kernel's output must be finite and not all zero, and for
/// `tiled` it must match the rows kernel on the same inputs ([`attn_parity`]):
/// a mismatch is an error, nothing more is timed.
fn run_attn_kernel(
    snap: &std::path::Path,
    kernel: EncoderAttnKernel,
    shapes: &[AttnShape],
    iters: usize,
    warmup: usize,
    reps: usize,
) -> Result<(), String> {
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let rt = GpuRuntime::new_inference()?;
    if kernel == EncoderAttnKernel::Tiled && !rt.has_tensorops() {
        return Err("--attn-kernel=tiled needs a device with TensorOps".into());
    }
    let mut kinds: Vec<(bool, u32, u32)> = Vec::new();
    for l in &cfg.layers {
        if !kinds.contains(&(l.sliding, l.head_dim, l.kv_heads)) {
            kinds.push((l.sliding, l.head_dim, l.kv_heads));
        }
    }
    println!(
        "encoder attention kernel {kernel:?}, heads {}, window {}",
        cfg.q_heads, cfg.sliding_window
    );
    for shape in shapes {
        let (b, t) = (shape.lens.len(), shape.t);
        for &(sliding, d, kv_heads) in &kinds {
            let (h, hkv, d) = (cfg.q_heads as usize, kv_heads as usize, d as usize);
            let dims = EncoderAttnDims {
                batch: b as u32,
                seq: t as u32,
                heads: cfg.q_heads,
                heads_kv: kv_heads,
                head_dim: d as u32,
                window: if sliding { cfg.sliding_window } else { 0 },
                scale: 1.0,
            };
            let mk = |n: usize, seed: u64| -> Result<tessl::tensor::GpuBuffer, String> {
                let buf = rt.alloc_buffer(n * 4)?;
                buf.write_f32(&common::fill_affine(n, seed, 1.0 / (d as f32).sqrt(), 0.0));
                Ok(buf)
            };
            let q = mk(b * t * h * d, 21)?;
            let k = mk(b * t * hkv * d, 22)?;
            let v = mk(b * t * hkv * d, 23)?;
            let o = rt.alloc_buffer(b * t * h * d * 4)?;
            let len_buf = rt.alloc_buffer(b * 4)?;
            len_buf.write_u32(&shape.lens.iter().map(|&l| l as u32).collect::<Vec<_>>());
            let out_of = |which: EncoderAttnKernel| -> Result<Vec<f32>, String> {
                o.write_f32(&vec![f32::NAN; b * t * h * d]);
                encoder_attn_with(&rt, &q, &k, &v, &o, &len_buf, dims, which)?;
                rt.synchronize()?;
                Ok(o.read_f32())
            };
            let what = format!("{} d={d}", shape.key);
            let out = out_of(kernel)?;
            if out.iter().any(|x| !x.is_finite()) || out.iter().all(|&x| x == 0.0) {
                return Err(format!("{what}: the output is not finite and non-zero"));
            }
            if kernel == EncoderAttnKernel::Tiled {
                let rows = out_of(EncoderAttnKernel::Rows)?;
                let err = attn_parity(&rows, &out, &shape.lens, t, h * d).map_err(|e| format!("{what}: {e}"))?;
                println!("parity {what}: tiled vs rows max abs error {err:.3e} (limit {ATTN_PARITY_REL:.0e} of the largest output)");
            }
            let mut ms = Vec::with_capacity(iters);
            for round in 0..warmup + iters {
                rt.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..reps {
                    encoder_attn_with(&rt, &q, &k, &v, &o, &len_buf, dims, kernel)?;
                }
                rt.synchronize()?;
                if round >= warmup {
                    ms.push(t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
                }
            }
            let med = common::median(ms)?;
            let kind = if sliding { "sliding" } else { "global" };
            println!("METRIC encoder_attn_{kind}_h{d}_{}_ms {med:.4}", shape.key);
        }
    }
    Ok(())
}

/// [`common::host_rms`] with a weight.
fn host_rms(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
    common::host_rms(x, Some(w), eps)
}

/// `--kernels`: the text tower's four fused row passes (`eg2t_resid_norm_f32`,
/// `eg2t_resid_norm_bf16`, `eg2t_resid_add_bf16_copy`, `eg2t_scale_rms_norm_f32`)
/// through their public entry points on random operands (no checkpoint, no
/// weights), each checked once against an f64 host reference before it is
/// timed: one `METRIC eg2t_{pass}_ms` per pass, the median of `BENCH_ITERS`
/// dispatches. The row passes are what the tower runs per layer; this lane
/// exists so `kernel_coverage` sees them dispatched.
fn run_text_kernels(warmup: usize, iters: usize) -> Result<(), String> {
    use common::{check_against, check_lane_output, poisoned_bf16, poisoned_f32, read_bf16, time_lane, upload_f32};
    use tessl::embedgemma2::{resid_add_bf16_copy, resid_add_norm_bf16, resid_add_norm_f32, scale_rms_norm_inplace};

    const ROWS: u32 = 5;
    const DIM: u32 = 96;
    const EPS: f32 = 1e-6;
    const LAYER_SCALE: f32 = 0.75;
    const SCALE: f32 = 9.5;
    let (rows, dim) = (ROWS as usize, DIM as usize);
    let n = rows * dim;
    let rt = GpuRuntime::new_inference()?;
    let y = common::fill_affine(n, 31, 3.0, 0.0);
    let resid0 = common::fill_affine(n, 32, 5.0, 0.0);
    let norm_w = |seed: u64| -> Vec<f32> { common::fill_affine(dim, seed, 0.4, 1.0) };
    let (w_post, w_next) = (norm_w(33), norm_w(34));
    let (y_b, w_post_b, w_next_b) = (
        upload_f32(&rt, &y)?,
        upload_f32(&rt, &w_post)?,
        upload_f32(&rt, &w_next)?,
    );

    // The post-norm residual add: `ls * (resid + rms(y) * w_post)` per row.
    let resid_after: Vec<f64> = (0..rows)
        .flat_map(|r| {
            let row = r * dim..(r + 1) * dim;
            let yn = host_rms(
                &y[row.clone()].iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                &w_post,
                f64::from(EPS),
            );
            let base = &resid0[row];
            (0..dim)
                .map(|i| f64::from(LAYER_SCALE) * (f64::from(base[i]) + yn[i]))
                .collect::<Vec<_>>()
        })
        .collect();
    let normed: Vec<f64> = (0..rows)
        .flat_map(|r| host_rms(&resid_after[r * dim..(r + 1) * dim], &w_next, f64::from(EPS)))
        .collect();

    for bf16_out in [false, true] {
        let lane = if bf16_out {
            "eg2t_resid_norm_bf16"
        } else {
            "eg2t_resid_norm_f32"
        };
        let resid = upload_f32(&rt, &resid0)?;
        let out = if bf16_out {
            poisoned_bf16(&rt, n)?
        } else {
            poisoned_f32(&rt, n)?
        };
        let dispatch = |resid: &tessl::tensor::GpuBuffer| {
            if bf16_out {
                resid_add_norm_bf16(
                    &rt,
                    &y_b,
                    &w_post_b,
                    resid,
                    &w_next_b,
                    &out,
                    ROWS,
                    DIM,
                    EPS,
                    LAYER_SCALE,
                )
            } else {
                resid_add_norm_f32(
                    &rt,
                    &y_b,
                    &w_post_b,
                    resid,
                    &w_next_b,
                    &out,
                    ROWS,
                    DIM,
                    EPS,
                    LAYER_SCALE,
                )
            }
        };
        dispatch(&resid)?;
        rt.synchronize()?;
        let got = if bf16_out {
            read_bf16(&rt, &out, n)?
        } else {
            out.read_f32()
        };
        check_lane_output(lane, &got)?;
        let (abs, rel) = if bf16_out { (1e-5, 1e-2) } else { (1e-5, 2e-4) };
        check_against(&format!("{lane} out"), &got[..n], &normed, abs, rel)?;
        check_against(
            &format!("{lane} resid"),
            &resid.read_f32()[..n],
            &resid_after,
            1e-5,
            2e-4,
        )?;
        // Timed on a residual that is refreshed by the dispatch itself; its
        // value drifts with the layer scale, which cannot change the time.
        time_lane(&rt, lane, warmup, iters, || dispatch(&resid))?;
    }

    let lane = "eg2t_resid_add_bf16_copy";
    let resid = upload_f32(&rt, &resid0)?;
    let copy = poisoned_bf16(&rt, n)?;
    resid_add_bf16_copy(&rt, &y_b, &w_post_b, &resid, &copy, ROWS, DIM, EPS, LAYER_SCALE)?;
    rt.synchronize()?;
    let got = read_bf16(&rt, &copy, n)?;
    check_lane_output(lane, &got)?;
    check_against(&format!("{lane} copy"), &got, &resid_after, 1e-5, 1e-2)?;
    check_against(
        &format!("{lane} resid"),
        &resid.read_f32()[..n],
        &resid_after,
        1e-5,
        2e-4,
    )?;
    time_lane(&rt, lane, warmup, iters, || {
        resid_add_bf16_copy(&rt, &y_b, &w_post_b, &resid, &copy, ROWS, DIM, EPS, LAYER_SCALE)
    })?;

    // The per-layer inputs' scale and norm: `rms(x * scale) * w`, in place.
    let lane = "eg2t_scale_rms_norm_f32";
    // Rows small enough that `eps` is a visible share of the mean square: rms(x * s) is
    // scale-invariant only while eps is negligible beside it, so with `|x| <= 2e-3` an
    // ignored (or doubled) scale moves the output by tens of percent and the check sees it.
    let y_small = common::fill_affine(n, 35, 2e-3, 0.0);
    let x = upload_f32(&rt, &y_small)?;
    scale_rms_norm_inplace(&rt, &x, &w_next_b, ROWS, DIM, EPS, SCALE)?;
    rt.synchronize()?;
    let want: Vec<f64> = (0..rows)
        .flat_map(|r| {
            let scaled: Vec<f64> = y_small[r * dim..(r + 1) * dim]
                .iter()
                .map(|&v| f64::from(v) * f64::from(SCALE))
                .collect();
            host_rms(&scaled, &w_next, f64::from(EPS))
        })
        .collect();
    let got = x.read_f32();
    check_lane_output(lane, &got[..n])?;
    check_against(lane, &got[..n], &want, 1e-5, 2e-4)?;
    time_lane(&rt, lane, warmup, iters, || {
        scale_rms_norm_inplace(&rt, &x, &w_next_b, ROWS, DIM, EPS, SCALE)
    })
}

fn main() -> Result<(), String> {
    if std::env::args().nth(1).as_deref() == Some("--kernels") {
        if let Some(extra) = std::env::args().nth(2) {
            return Err(format!("unexpected argument {extra:?} after --kernels"));
        }
        let iters = env_usize("BENCH_ITERS", 10, 1)?;
        let warmup = env_usize("BENCH_WARMUP", 3, 0)?;
        let outcome = common::mark_incomplete_on_error(run_text_kernels(warmup, iters));
        common::emit_kernel_trace();
        return outcome;
    }
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let mut args = std::env::args().skip(1);
    if let Some(arg) = args.next() {
        let kernel = match arg.strip_prefix("--attn-kernel=") {
            Some("rows") => EncoderAttnKernel::Rows,
            Some("tiled") => EncoderAttnKernel::Tiled,
            _ => return Err(format!("expected --attn-kernel=rows|tiled or nothing, got {arg:?}")),
        };
        if let Some(extra) = args.next() {
            return Err(format!("unexpected argument {extra:?}"));
        }
        // Everything that can be refused is, before the runtime exists or any
        // METRIC line is printed.
        if std::env::var_os("BENCH_WORKLOADS").is_some() {
            return Err("BENCH_WORKLOADS does not apply to --attn-kernel; its shapes are BENCH_ATTN_SHAPES".into());
        }
        let spec = match std::env::var("BENCH_ATTN_SHAPES") {
            Ok(v) => v,
            Err(std::env::VarError::NotPresent) => DEFAULT_ATTN_SHAPES.into(),
            Err(e) => return Err(format!("BENCH_ATTN_SHAPES: {e}")),
        };
        let shapes = parse_attn_shapes(&spec)?;
        let iters = env_usize("BENCH_ITERS", 10, 1)?;
        let warmup = env_usize("BENCH_WARMUP", 3, 0)?;
        let reps = env_usize("BENCH_ATTN_REPS", 8, 1)?;
        return common::mark_incomplete_on_error(run_attn_kernel(&snap, kernel, &shapes, iters, warmup, reps));
    }
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    let workloads = spec
        .split(',')
        .map(|w| Ok((w.trim().to_string(), parse_workload(w)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;

    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    let async_encode = match std::env::var("BENCH_ASYNC").as_deref() {
        Err(_) | Ok("0") => false,
        Ok("1") => true,
        Ok(other) => return Err(format!("BENCH_ASYNC={other:?}: expected 0 or 1")),
    };
    rt.set_async_encode(async_encode)?;
    let serialized = match std::env::var("BENCH_SERIALIZED").as_deref() {
        Err(_) | Ok("0") => false,
        Ok("1") => true,
        Ok(other) => return Err(format!("BENCH_SERIALIZED={other:?}: expected 0 or 1")),
    };
    rt.set_serialized_dispatch(serialized)?;
    let (operands, backend) = match std::env::var("BENCH_PRECISION").as_deref() {
        Err(_) | Ok("f32") => (GemmOperands::ExactF32, "tessl-f32"),
        Ok("bf16") => (GemmOperands::Bf16, "tessl-bf16"),
        Ok(other) => return Err(format!("BENCH_PRECISION={other:?}: expected f32 or bf16")),
    };
    let mut model = EmbedGemma2Model::load_with(&rt, &st, "language_model.", cfg.clone(), operands)?;
    if let Ok(v) = std::env::var("BENCH_PLE_RUN_BYTES") {
        let n: usize = v.parse().map_err(|e| format!("BENCH_PLE_RUN_BYTES={v:?}: {e}"))?;
        model.set_ple_run_bytes(n)?;
    }
    let weight_mib = model.weight_bytes() as f64 / MIB;
    let dim = cfg.embedding_dim as usize;
    let loaded = rt.current_allocated_bytes();

    let mut rows = Vec::new();
    for (name, lens) in &workloads {
        let seqs: Vec<Vec<u32>> = lens.iter().enumerate().map(|(s, &t)| ids(s, t)).collect();
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();

        rt.reset_peak_allocated_bytes();
        let out = model.encode(&batch, None, false)?;
        for (s, e) in out.embeddings.chunks(dim).enumerate() {
            let norm = e.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
            if !e.iter().all(|v| v.is_finite()) || (norm - 1.0).abs() > 1e-4 {
                return Err(format!(
                    "{name}: sequence {s} embedding is not a finite unit vector (norm {norm})"
                ));
            }
        }

        for _ in 0..warmup {
            model.encode(&batch, None, false)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.encode(&batch, None, false)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        let device_peak = rt.peak_allocated_bytes().saturating_sub(loaded);
        rows.push(format!(
            "{{\"workload\":\"{name}\",\"backend\":\"{backend}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"async\":{},\"serialized\":{},\"forwards\":{},\"weight_mib\":{weight_mib:.1},\"device_peak_mib\":{:.1},\"peak_footprint_mib\":{:.1}}}",
            ms[0],
            ms[ms.len() / 2],
            u8::from(async_encode),
            u8::from(serialized),
            out.forwards,
            device_peak as f64 / MIB,
            peak_footprint()? as f64 / MIB,
        ));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{attn_parity, parse_attn_shapes, parse_workload, DEFAULT_ATTN_SHAPES};

    #[test]
    fn workload_grammar() {
        assert_eq!(parse_workload("3x5").unwrap(), vec![5, 5, 5]);
        assert_eq!(parse_workload("1x9+2x4").unwrap(), vec![9, 4, 4]);
        // The i-th length is LO + (i * 2654435761) % (HI - LO + 1), which
        // bench/embedgemma2_torch.py repeats.
        assert_eq!(parse_workload("4x10-42").unwrap(), vec![10, 20, 30, 40]);
        for bad in ["0x5", "2x1", "2x9-3", "2", "x5", "2x5+"] {
            assert!(parse_workload(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn default_attn_shapes_parse_and_include_padding() {
        let shapes = parse_attn_shapes(DEFAULT_ATTN_SHAPES).unwrap();
        assert!(
            shapes.iter().any(|s| s.lens.iter().any(|&l| l < s.t)),
            "no default shape has a row shorter than T: {shapes:?}"
        );
        let keys: Vec<&str> = shapes.iter().map(|s| s.key.as_str()).collect();
        assert!(keys.contains(&"b64_t32") && keys.contains(&"b1_t6147"), "{keys:?}");
    }

    #[test]
    fn attn_shapes_refuse_any_bad_entry_and_repeats() {
        for bad in ["", "64x32,", "0x5", "64x32,2", "3x5,3x5", "2x9-3"] {
            assert!(parse_attn_shapes(bad).is_err(), "{bad:?}");
        }
        let ragged = parse_attn_shapes("4x10-42").unwrap();
        assert_eq!((ragged[0].t, ragged[0].key.as_str()), (40, "b4_t40_len100"));
    }

    #[test]
    fn parity_accepts_a_match_and_refuses_each_kind_of_mismatch() {
        // 2 sequences of 3 positions, 2 floats per position; sequence 1 has 2 live positions.
        let lens = [3, 2];
        let want: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0];
        assert!(attn_parity(&want, &want, &lens, 3, 2).unwrap() == 0.0);
        let near: Vec<f32> = want.iter().map(|x| x * (1.0 + 1e-6)).collect();
        assert!(attn_parity(&want, &near, &lens, 3, 2).is_ok());
        let mut far = want.clone();
        far[0] += 0.01;
        assert!(attn_parity(&want, &far, &lens, 3, 2).unwrap_err().contains("differs"));
        let mut pad = want.clone();
        pad[10] = 1e-9;
        assert!(attn_parity(&want, &pad, &lens, 3, 2).unwrap_err().contains("padding"));
        let mut nan = want.clone();
        nan[3] = f32::NAN;
        assert!(attn_parity(&want, &nan, &lens, 3, 2)
            .unwrap_err()
            .contains("not finite"));
        assert!(attn_parity(&want, &want[1..], &lens, 3, 2)
            .unwrap_err()
            .contains("lengths"));
        assert!(attn_parity(&[0.0; 12], &[0.0; 12], &[3, 3], 3, 2)
            .unwrap_err()
            .contains("all zero"));
    }
}
