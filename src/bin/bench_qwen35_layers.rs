//! Prefill and decode throughput of `tessl::qwen35_model::Qwen35Model` at
//! Qwen3.5-2B's shapes, and the paired single-kernel lanes.
//!
//! ```text
//! cargo run --release --bin bench_qwen35_layers                 # T = 1024, 2048, 8192
//! cargo run --release --bin bench_qwen35_layers -- 1024 4096    # chosen T
//! cargo run --release --bin bench_qwen35_layers -- --check-only # gate, no timing
//! cargo run --release --bin bench_qwen35_layers -- --weights=DIR 8192   # real checkpoint
//! cargo run --release --bin bench_qwen35_layers -- --gdn-scan32 # the other GDN scan slice
//! cargo run --release --bin bench_qwen35_layers -- --iters=11   # samples per timing
//! cargo run --release --bin bench_qwen35_layers -- --decode=4096  # prefill + per-token decode
//! cargo run --release --bin bench_qwen35_layers -- --model-forward 1024 2048  # METRIC lines only
//! cargo run --release --bin bench_qwen35_layers -- --paired-attn 200 8192
//! cargo run --release --bin bench_qwen35_layers -- --attn-tiles 2048  # METRIC per tile
//! cargo run --release --bin bench_qwen35_layers -- --paired-gdn-recurrent 1 4
//! cargo run --release --bin bench_qwen35_layers -- --paired-gate-up 512 2048
//! ```
//!
//! The model under test is `Qwen35Model` itself, so the forward timed is the
//! one `tests/qwen35_model.rs` checks against transformers: the layer order is
//! written once, in `Qwen35Model::layer`, and this bench wires no layer of its
//! own. Weights are `Qwen35Model::random` (seeded, bf16, with the packed LM head)
//! unless `--weights=DIR` names a Qwen3.5 checkpoint directory (its
//! `config.json` and the one `*.safetensors` file in it, tensors under
//! `model.language_model.`), which is loaded at bf16 with the head. The GDN
//! scan uses the library's default slice (`GdnScanSlice::default()`, Cols16);
//! `--gdn-scan32` selects the other. Both are bit-identical.
//!
//! What is timed, per T, with async encode (one command buffer per forward),
//! `--iters` samples (default 7) after 2 warm-ups, as min and median:
//!
//! * **tower** — `begin` + `advance_to(all layers)` + `final_norm_f32`: the
//!   prefill without the LM head, the figure earlier reports called the layer
//!   bench's forward (`tower tok/s`). Also printed as `METRIC
//!   model_forward_ms_t{T}` (median), the name `bench_paired --format metrics` users
//!   compare across builds.
//! * **layers** — the same with `begin` outside the timed span: all layers and
//!   the final norm on preallocated activations, comparable to the earlier
//!   bench's "forward" column.
//! * **forward_rows(Last)** — the tower plus the LM head for the last row.
//! * **prefill** — `Qwen35Model::prefill(ids, 1)` to the first logits: the
//!   tower, the head and the state a decode continues from.
//! * **per layer / per part** — `Staged::encode_part`, `REPS` per command
//!   buffer so the ~0.25 ms submit-and-wait floor (`docs/benchmarking.md`) is
//!   not counted once per part. A GDN or attention layer is its input norm and
//!   mixer; the MLP is its post-norm and SwiGLU MLP (`LayerPart`). Summed layers
//!   are printed next to the measured layers figure as a cross-check. The
//!   mixer's inner stages (in-projection, conv, scan, ...) are not split out:
//!   the model has no stage seam below `LayerPart`.
//!
//! Before anything is timed, one forward runs and its final norm and logits
//! must be finite and not all zero; there is no timing without it.
//!
//! `--decode[=P]` times batch-1 decode after a prefill of `P` positions
//! (default 4096): `Qwen35Model::prefill`, then `Decode::step` per token (a
//! fixed token pattern, not sampled), tok/s as min and median over the
//! steps, and `Decode::step`'s per-token host waits, command buffer commits,
//! dispatches and cold allocations from `infer_trace` over separate traced
//! steps: the baseline for a GPU-resident decode loop. Every figure is a
//! `METRIC` line for `bench_paired --format metrics`. Each step is the token's layers,
//! the LM head and the 1 MB logits read; the sampler is not included
//! (`bench_nn_kernels` times the samplers).
//!
//! `--paired-attn`, `--paired-gdn-recurrent` and `--paired-gate-up` time one
//! kernel or kernel pair against another in this process (ABBA order, which
//! goes first rotating each round); none of them composes layers.
//! `--paired-attn` times the prefill attention kernels on the same Q/K/V.
//! `--paired-gate-up` times the gate and up projections as the training
//! forward runs them on bf16 storage (two GEMMs, `[T, 2048] x [2048, 6144]`
//! each, bf16 operands into f32) against one packed GEMM (`[T, 2048] x [2048,
//! 12288]`), in both encode modes: async (16 units per command buffer) and
//! sync, the runtime's default and ojas-qwen35's training mode. Default T is
//! 512 and 2048.
//!
//! `--attn-tiles [T...]` times `qwen35::attn_prefill_with_tile` at each of the
//! four tiles on the same Q/K/V (default T = 2048) and prints one `METRIC
//! attn_prefill_q{BQ}_k{BK}_sg{NSG}_ms_t{T}` per tile: the lane
//! `bench_paired --format metrics` alternates between two builds of this binary.
//!
//! `--model-forward [T...]` runs the tower lane alone and prints its METRIC
//! lines. Default T is 1024 and 2048.
//!
//! Earlier versions of this bench built their own 24 layers from the kernels
//! and had `--attn-rows`, `--attn-tile=`, `--mlp-unfused` and `--gdn-scan16`
//! switches that swapped kernels inside that wiring; the model has no such
//! knobs, so they are gone (`--paired-attn` and `bench/results/
//! qwen35_fused_swiglu_gemm_m5pro.txt` hold those comparisons).

mod common;

use std::sync::Arc;
use std::time::Instant;

use tessl::qwen35::{self, Cols, GdnDims, GdnParams, GdnProjLayout, StateIn};
use tessl::tensor::{f32_slice_to_bf16, GpuBuffer};
use tessl::{gemm, nn, DType, GemmBackend, GpuRuntime, Tensor};

// Qwen3.5-2B `text_config` (models--Qwen--Qwen3.5-2B-Base config.json).
const HIDDEN: usize = 2048;
const INTER: usize = 6144;
const GDN_K_HEADS: u32 = 16;
const GDN_V_HEADS: u32 = 16;
const GDN_V_DIM: u32 = 128;
const Q_HEADS: u32 = 8;
const KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 256;

const BACKEND: GemmBackend = GemmBackend::TensorOps;
const REPS: usize = 8;
const WARMUP: usize = 2;
const ITERS: usize = 7;
/// Upper bound on `--iters`: keeps the sample buffers and the run time sane.
const MAX_ITERS: usize = 10_000;
const DEFAULT_T: [usize; 3] = [1024, 2048, 8192];

type Res<T> = Result<T, String>;

fn buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> Res<GpuBuffer> {
    let b = rt.alloc_buffer(data.len().max(1) * 4)?;
    b.write_f32(data);
    Ok(b)
}

fn buf_u32(rt: &Arc<GpuRuntime>, data: &[u32]) -> Res<GpuBuffer> {
    let b = rt.alloc_buffer(data.len().max(1) * 4)?;
    b.write_u32(data);
    Ok(b)
}

/// A bf16 weight `[rows, cols]` uploaded from shared host bits: distinct device
/// memory per layer without regenerating a billion random numbers.
fn weight(rt: &Arc<GpuRuntime>, rows: usize, cols: usize, bits: &[u16]) -> Res<Tensor> {
    let t = rt.alloc_tensor_bf16(&[rows, cols])?;
    t.buffer.write_bf16_bits(bits);
    Ok(t)
}

/// Uniform in ±sqrt(3 / fan_in): unit-variance outputs from unit-variance inputs.
fn weight_bits(rows: usize, cols: usize, seed: u64) -> Vec<u16> {
    f32_slice_to_bf16(&common::fill_affine(rows * cols, seed, (3.0 / rows as f32).sqrt(), 0.0))
}

// --------------------------------------------------------------------- gate ---

fn finite_f32(name: &str, b: &GpuBuffer, n: usize) -> Res<()> {
    let v = b.read_f32();
    match v[..n].iter().position(|x| !x.is_finite()) {
        Some(i) => Err(format!(
            "{name}: element {i} of {n} is {} (unwritten or overflowed)",
            v[i]
        )),
        None => Ok(()),
    }
}

// ------------------------------------------------------------------- timing ---

/// Median milliseconds of one `f`, with `reps` of it per command buffer.
fn time_ms(rt: &Arc<GpuRuntime>, reps: usize, iters: usize, mut f: impl FnMut() -> Res<()>) -> Res<f64> {
    rt.synchronize()?;
    for _ in 0..WARMUP {
        for _ in 0..reps {
            f()?;
        }
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..reps {
            f()?;
        }
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
    }
    common::median(samples)
}

// ------------------------------------------------------- paired attention ---

/// Dispatches per command buffer. T = 200 is ~0.16 GFLOP, under the ~0.25 ms
/// submit floor if timed one launch at a time (`docs/benchmarking.md`). The
/// count keeps that buffer well above the floor. T = 8192 is ~275 GFLOP, so
/// a handful of launches is enough and the slow kernel stays a short sample.
fn paired_reps(t: usize) -> usize {
    if t <= 512 {
        128
    } else if t <= 2048 {
        16
    } else {
        4
    }
}

#[derive(Clone, Copy)]
enum PairKernel {
    Rows,
    Tiled(qwen35::AttnTile),
}

impl PairKernel {
    fn name(self) -> &'static str {
        match self {
            PairKernel::Rows => "flash_attn_rows",
            PairKernel::Tiled(_) => "attn_prefill",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_pair(
    rt: &Arc<GpuRuntime>,
    kernel: PairKernel,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    pos: &GpuBuffer,
    t: usize,
) -> Res<()> {
    let dims = nn::AttnDims {
        batch: 1,
        tq: t as u32,
        heads: Q_HEADS,
        heads_kv: KV_HEADS,
        window: 0,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };
    match kernel {
        PairKernel::Rows => nn::flash_attn_rows(rt, q, k, v, o, tkv, pos, pos, dims, HEAD_DIM, false),
        PairKernel::Tiled(tile) => qwen35::attn_prefill_with_tile(rt, q, k, v, o, tkv, pos, pos, dims, false, tile),
    }
}

/// One attention layer at Qwen3.5-2B's head geometry, both prefill kernels,
/// same buffers, ABBA. Prints median and min milliseconds per kernel per T.
fn run_paired_attn(rt: &Arc<GpuRuntime>, ts: &[usize]) -> Res<()> {
    use std::io::Write;

    const WARMUP_ROUNDS: usize = 3;
    const ROUNDS: usize = 9;
    println!(
        "paired prefill attention: nn::flash_attn_rows vs qwen35::attn_prefill \
         (tile {})",
        qwen35::ATTN_PREFILL_TILE.label()
    );
    println!(
        "one process, one attention layer, batch 1, heads {Q_HEADS}q/{KV_HEADS}kv x {HEAD_DIM}, \
         same Q/K/V, ABBA (first kernel rotates each round)"
    );
    println!("warmup rounds {WARMUP_ROUNDS}, timed rounds {ROUNDS} (one sample per kernel per round)");

    for &t in ts {
        if t > u32::MAX as usize {
            return Err(format!("T={t} does not fit the kernel's u32 length"));
        }
        let reps = paired_reps(t);
        let qd = t * (Q_HEADS * HEAD_DIM) as usize;
        let kv = t * (KV_HEADS * HEAD_DIM) as usize;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let q = buf(rt, &common::fill_affine(qd, 11, scale, 0.0))?;
        let k = buf(rt, &common::fill_affine(kv, 12, scale, 0.0))?;
        let v = buf(rt, &common::fill_affine(kv, 13, scale, 0.0))?;
        let o = rt.alloc_buffer(qd * 4)?;
        let tkv = buf_u32(rt, &[t as u32])?;
        let pos = buf_u32(rt, &[0])?;

        let mut checksums = Vec::new();
        for kernel in [PairKernel::Rows, PairKernel::Tiled(qwen35::ATTN_PREFILL_TILE)] {
            o.write_f32(&vec![f32::NAN; qd]);
            dispatch_pair(rt, kernel, &q, &k, &v, &o, &tkv, &pos, t)?;
            rt.synchronize()?;
            let out = o.read_f32();
            if let Some(i) = out[..qd].iter().position(|x| !x.is_finite()) {
                return Err(format!("T={t} {}: element {i} of {qd} is non-finite", kernel.name()));
            }
            let sum: f64 = out[..qd].iter().map(|x| x.abs() as f64).sum();
            if sum == 0.0 {
                return Err(format!("T={t} {}: output is all zeros", kernel.name()));
            }
            checksums.push((kernel.name(), sum));
        }
        let rel = (checksums[0].1 - checksums[1].1).abs() / checksums[0].1;
        println!(
            "T={t}: wrote finite output, |checksum rows-tiled|/rows = {rel:.3e} \
             (rows {:.6e}, tiled {:.6e})",
            checksums[0].1, checksums[1].1
        );
        if rel > 1e-3 {
            return Err(format!(
                "T={t}: kernels disagree by {rel:.3e} on the abs checksum, above 1e-3"
            ));
        }

        let mut rows = Vec::with_capacity(ROUNDS);
        let mut tiled = Vec::with_capacity(ROUNDS);
        for round in 0..(WARMUP_ROUNDS + ROUNDS) {
            let record = round >= WARMUP_ROUNDS;
            let order = if round % 2 == 0 {
                [PairKernel::Rows, PairKernel::Tiled(qwen35::ATTN_PREFILL_TILE)]
            } else {
                [PairKernel::Tiled(qwen35::ATTN_PREFILL_TILE), PairKernel::Rows]
            };
            for kernel in order {
                rt.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..reps {
                    dispatch_pair(rt, kernel, &q, &k, &v, &o, &tkv, &pos, t)?;
                }
                rt.synchronize()?;
                let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
                if record {
                    let which = if round % 2 == 0 { "rows-first" } else { "tiled-first" };
                    println!(
                        "sample T={t} round={} {which} {} {ms:.4} ms (buffer {:.2} ms, {reps} launches)",
                        round - WARMUP_ROUNDS,
                        kernel.name(),
                        ms * reps as f64
                    );
                    let _ = std::io::stdout().flush();
                    match kernel {
                        PairKernel::Rows => rows.push(ms),
                        PairKernel::Tiled(_) => tiled.push(ms),
                    }
                }
            }
        }

        for (name, samples) in [("flash_attn_rows", &rows), ("attn_prefill", &tiled)] {
            let med = common::median(samples.clone())?;
            let lo = samples.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "PAIRED t={t} kernel={name} n={} reps={reps} median_ms={med:.4} min_ms={lo:.4} max_ms={hi:.4} buffer_median_ms={:.2}",
                samples.len(),
                med * reps as f64
            );
        }
        let ratio = common::median(tiled.clone())? / common::median(rows.clone())?;
        println!("PAIRED t={t} ratio_tiled_over_rows={ratio:.4}");
    }
    Ok(())
}

/// `--attn-tiles`: every `qwen35_attn_tiled_h256_*` geometry on the same Q/K/V
/// at the 2B's head geometry, one process, the tile order rotating each round
/// (every tile meets every position in the round). Prints one `METRIC
/// attn_prefill_{label}_ms_t{T}` (the median over rounds, per launch) per tile
/// per T, so `bench_paired --format metrics` alternates whole builds of this
/// binary round by round.
fn run_attn_tiles(rt: &Arc<GpuRuntime>, ts: &[usize]) -> Res<()> {
    const WARMUP_ROUNDS: usize = 3;
    const ROUNDS: usize = 8;
    let tiles = qwen35::AttnTile::ALL;
    println!(
        "attn_prefill_with_tile, every tile: batch 1, heads {Q_HEADS}q/{KV_HEADS}kv x {HEAD_DIM}, causal from 0; \
         tile order rotates each round; {WARMUP_ROUNDS} warmup + {ROUNDS} timed rounds"
    );
    for &t in ts {
        if t > u32::MAX as usize {
            return Err(format!("T={t} does not fit the kernel's u32 length"));
        }
        let reps = paired_reps(t);
        let qd = t * (Q_HEADS * HEAD_DIM) as usize;
        let kv = t * (KV_HEADS * HEAD_DIM) as usize;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let q = buf(rt, &common::fill_affine(qd, 11, scale, 0.0))?;
        let k = buf(rt, &common::fill_affine(kv, 12, scale, 0.0))?;
        let v = buf(rt, &common::fill_affine(kv, 13, scale, 0.0))?;
        let o = rt.alloc_buffer(qd * 4)?;
        let tkv = buf_u32(rt, &[t as u32])?;
        let pos = buf_u32(rt, &[0])?;

        for tile in tiles {
            o.write_f32(&vec![f32::NAN; qd]);
            dispatch_pair(rt, PairKernel::Tiled(tile), &q, &k, &v, &o, &tkv, &pos, t)?;
            rt.synchronize()?;
            let out = o.read_f32();
            if let Some(i) = out[..qd].iter().position(|x| !x.is_finite()) {
                return Err(format!("T={t} {}: element {i} of {qd} is non-finite", tile.label()));
            }
            if out[..qd].iter().all(|&x| x == 0.0) {
                return Err(format!("T={t} {}: output is all zeros", tile.label()));
            }
        }

        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); tiles.len()];
        for round in 0..(WARMUP_ROUNDS + ROUNDS) {
            for step in 0..tiles.len() {
                let i = (round + step) % tiles.len();
                rt.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..reps {
                    dispatch_pair(rt, PairKernel::Tiled(tiles[i]), &q, &k, &v, &o, &tkv, &pos, t)?;
                }
                rt.synchronize()?;
                if round >= WARMUP_ROUNDS {
                    samples[i].push(t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
                }
            }
        }
        for (tile, s) in tiles.iter().zip(samples) {
            let lo = s.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let med = common::median(s)?;
            println!(
                "attn_prefill {} T={t}: median {med:.4} ms, min {lo:.4}, max {hi:.4} ({reps} launches per buffer)",
                tile.label()
            );
            println!("METRIC attn_prefill_{}_ms_t{t} {med:.4}", tile.label());
        }
    }
    Ok(())
}

// ----------------------------------------------------- paired GDN recurrent ---

/// `qwen35::gdn_recurrent` launches per command buffer in `--paired-gdn-recurrent`:
/// one decode token is a few microseconds, so a buffer of these is well over
/// the submit floor.
const PAIRED_REC_REPS: usize = 256;

fn dispatch_rec(rt: &Arc<GpuRuntime>, m: &PairRec, slice: qwen35::GdnScanSlice) -> Res<()> {
    qwen35::gdn_recurrent_with_slice(
        rt,
        &m.dims,
        &m.gdn.conv_qkv(&m.qkv),
        &m.gdn.gates(&m.proj),
        &GdnParams {
            a_log: &m.a_log,
            dt_bias: &m.dt_bias,
        },
        StateIn::Snapshot(&m.state),
        Cols::dense(&m.out, m.gdn.value_dim()),
        Some(&m.state_out),
        slice,
    )
}

/// One GDN layer's decode operands at 2B shapes: a read-only snapshot state,
/// so every launch computes the same thing.
struct PairRec {
    gdn: GdnProjLayout,
    dims: GdnDims,
    proj: GpuBuffer,
    qkv: GpuBuffer,
    a_log: GpuBuffer,
    dt_bias: GpuBuffer,
    state: GpuBuffer,
    out: GpuBuffer,
    state_out: GpuBuffer,
}

/// `gdn_recurrent` in 32- and 16-column slices at decode (T = 1), same
/// operands, ABBA, per batch size. Checks the two agree bit for bit first.
fn run_paired_gdn_recurrent(rt: &Arc<GpuRuntime>, batches: &[usize]) -> Res<()> {
    use qwen35::GdnScanSlice::{Cols16, Cols32};
    const WARMUP_ROUNDS: usize = 3;
    const ROUNDS: usize = 15;
    let gdn = GdnProjLayout::new(GDN_K_HEADS, GDN_V_HEADS, GDN_V_DIM)?;
    println!(
        "paired gdn_recurrent: 32- vs 16-column slices, T = 1, GDN {GDN_K_HEADS}k/{GDN_V_HEADS}v heads x \
         {GDN_V_DIM}, snapshot state; ABBA, {PAIRED_REC_REPS} launches per command buffer, \
         {WARMUP_ROUNDS} warmup + {ROUNDS} timed rounds"
    );
    for &b in batches {
        let dims = gdn.dims(b as u32, 1);
        let vh = GDN_V_HEADS as usize;
        let m = PairRec {
            gdn,
            dims,
            proj: buf(rt, &common::fill_affine(b * gdn.width() as usize, 31, 1.0, 0.0))?,
            qkv: buf(rt, &common::fill_affine(b * gdn.conv_dim() as usize, 32, 1.0, 0.0))?,
            a_log: buf(
                rt,
                &common::fill_affine(vh, 33, 1.0, 0.0)
                    .iter()
                    .map(|v| 1.4 + 1.3 * v)
                    .collect::<Vec<_>>(),
            )?,
            dt_bias: buf(
                rt,
                &common::fill_affine(vh, 34, 1.0, 0.0)
                    .iter()
                    .map(|v| -4.5 + 2.5 * v)
                    .collect::<Vec<_>>(),
            )?,
            state: buf(rt, &common::fill_affine(dims.state_elems_per_row(), 35, 0.05, 0.0))?,
            out: rt.alloc_buffer(b * gdn.value_dim() as usize * 4)?,
            state_out: rt.alloc_buffer(b * dims.state_elems_per_row() * 4)?,
        };
        let mut bits = Vec::new();
        for slice in [Cols32, Cols16] {
            m.out.write_f32(&vec![f32::NAN; m.out.nbytes() / 4]);
            m.state_out.write_f32(&vec![f32::NAN; m.state_out.nbytes() / 4]);
            dispatch_rec(rt, &m, slice)?;
            rt.synchronize()?;
            finite_f32("gdn_recurrent out", &m.out, m.out.nbytes() / 4)?;
            let mut v = m.out.read_u32();
            v.extend(m.state_out.read_u32());
            bits.push(v);
        }
        if bits[0] != bits[1] {
            return Err(format!(
                "B={b}: the 16-column gdn_recurrent differs from the 32-column one"
            ));
        }
        println!("B={b}: 16- and 32-column outputs and states are bit-identical");

        let (mut t32, mut t16, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
        for round in 0..(WARMUP_ROUNDS + ROUNDS) {
            let order = if round % 2 == 0 {
                [Cols32, Cols16]
            } else {
                [Cols16, Cols32]
            };
            let mut pair = [0.0f64; 2];
            for slice in order {
                rt.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..PAIRED_REC_REPS {
                    dispatch_rec(rt, &m, slice)?;
                }
                rt.synchronize()?;
                pair[usize::from(slice == Cols16)] = t0.elapsed().as_secs_f64() * 1e6 / PAIRED_REC_REPS as f64;
            }
            if round >= WARMUP_ROUNDS {
                t32.push(pair[0]);
                t16.push(pair[1]);
                ratios.push(pair[1] / pair[0]);
            }
        }
        let lo = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for (name, v) in [("cols32", &t32), ("cols16", &t16)] {
            println!(
                "PAIRED b={b} slice={name} median_us={:.3} min_us={:.3} max_us={:.3}",
                common::median(v.clone())?,
                lo(v),
                hi(v)
            );
        }
        println!(
            "PAIRED b={b} ratio_cols16_over_cols32 median={:.4} min={:.4} max={:.4}",
            common::median(ratios.clone())?,
            lo(&ratios),
            hi(&ratios)
        );
        println!("METRIC gdn_rec_b{b}_cols32_us {:.4}", common::median(t32)?);
        println!("METRIC gdn_rec_b{b}_cols16_us {:.4}", common::median(t16)?);
        println!("METRIC gdn_rec_b{b}_ratio_16_over_32 {:.4}", common::median(ratios)?);
    }
    Ok(())
}

// ------------------------------------------------------------ paired lanes ---

/// The two lanes of a paired comparison: the production path (`Prod`) and
/// the candidate (`Cand`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    Prod,
    Cand,
}

/// [`paired_lanes`]' lane labels for `--paired-gate-up`.
const GATE_UP_LANES: [&str; 2] = ["two_gemms", "packed_gemm"];

/// `((median, min) of a, (median, min) of b, median ratio)` of [`paired_summary`].
type PairedSummary = ((f64, f64), (f64, f64), f64);

/// `(median, min)` of `samples`, and the median of the per-round ratios
/// `b[i] / a[i]` (each round times both lanes back to back, so a round's
/// ratio cancels the drift between rounds).
fn paired_summary(a: &[f64], b: &[f64]) -> Res<PairedSummary> {
    let stats = |v: &[f64]| -> Res<(f64, f64)> {
        Ok((
            common::median(v.to_vec())?,
            v.iter().copied().fold(f64::INFINITY, f64::min),
        ))
    };
    let ratios: Vec<f64> = a.iter().zip(b).map(|(x, y)| y / x).collect();
    Ok((stats(a)?, stats(b)?, common::median(ratios)?))
}

/// Time `run(lane)` for both lanes, `reps` per command buffer, ABBA over
/// `rounds`, and print one PAIRED line per lane (labelled `labels`) plus the
/// ratio of the candidate to the production lane.
#[allow(clippy::too_many_arguments)]
fn paired_lanes(
    rt: &Arc<GpuRuntime>,
    t: usize,
    unit: &str,
    labels: [&str; 2],
    reps: usize,
    (warmup, rounds): (usize, usize),
    mut run: impl FnMut(Lane) -> Res<()>,
) -> Res<()> {
    let (mut a, mut b) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
    for round in 0..(warmup + rounds) {
        let order = if round % 2 == 0 {
            [Lane::Prod, Lane::Cand]
        } else {
            [Lane::Cand, Lane::Prod]
        };
        for lane in order {
            rt.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..reps {
                run(lane)?;
            }
            rt.synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            if round >= warmup {
                match lane {
                    Lane::Prod => a.push(ms),
                    Lane::Cand => b.push(ms),
                }
            }
        }
    }
    let ((a_med, a_min), (b_med, b_min), ratio) = paired_summary(&a, &b)?;
    for (label, med, min, v) in [(labels[0], a_med, a_min, &a), (labels[1], b_med, b_min, &b)] {
        let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "PAIRED t={t} unit={unit} lane={label} n={} reps={reps} median_ms={med:.4} min_ms={min:.4} max_ms={max:.4}",
            v.len()
        );
    }
    println!(
        "PAIRED t={t} unit={unit} ratio_{}_over_{} median_of_round_ratios={ratio:.4} \
         of_medians={:.4} of_mins={:.4}",
        labels[1],
        labels[0],
        b_med / a_med,
        b_min / a_min
    );
    Ok(())
}

/// Every element of the first `n` of `t` (f32 or bf16) finite, and not all zero.
fn finite_nonzero(what: &str, t: &Tensor, n: usize) -> Res<()> {
    let v: Vec<f32> = match t.dtype {
        DType::BF16 => t.buffer.contents_u16()[..n]
            .iter()
            .map(|&b| tessl::tensor::bf16_bits_to_f32(b))
            .collect(),
        _ => t.buffer.read_f32()[..n].to_vec(),
    };
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(format!("{what}: element {i} of {n} is non-finite"));
    }
    if v.iter().all(|&x| x == 0.0) {
        return Err(format!("{what}: all zeros"));
    }
    Ok(())
}

// --------------------------------------------------------- paired gate/up ---

/// `--paired-gate-up`: the gate and up projections as two GEMMs
/// ([`Lane::Prod`]) or one packed GEMM ([`Lane::Cand`]).
fn run_paired_gate_up(rt: &Arc<GpuRuntime>, ts: &[usize]) -> Res<()> {
    const ROUNDS: (usize, usize) = (3, 15);
    println!("paired gate/up: two GEMMs [T,2048]x[2048,6144] vs one packed [T,2048]x[2048,12288]");
    println!(
        "bf16 operands into f32 C, as GemmOperands::Bf16 runs them; ABBA, {} warmup + {} timed rounds",
        ROUNDS.0, ROUNDS.1
    );
    let gate_w = weight(rt, HIDDEN, INTER, &weight_bits(HIDDEN, INTER, 51))?;
    let up_w = weight(rt, HIDDEN, INTER, &weight_bits(HIDDEN, INTER, 52))?;
    // `gate | up` side by side, as a packed layout would store them.
    let packed_w = {
        let (g, u) = (weight_bits(HIDDEN, INTER, 51), weight_bits(HIDDEN, INTER, 52));
        let mut bits = Vec::with_capacity(HIDDEN * 2 * INTER);
        for r in 0..HIDDEN {
            bits.extend_from_slice(&g[r * INTER..(r + 1) * INTER]);
            bits.extend_from_slice(&u[r * INTER..(r + 1) * INTER]);
        }
        weight(rt, HIDDEN, 2 * INTER, &bits)?
    };
    for &t in ts {
        let x = rt.alloc_tensor_bf16(&[t, HIDDEN])?;
        x.buffer
            .write_bf16_bits(&f32_slice_to_bf16(&common::fill_affine(t * HIDDEN, 53, 1.0, 0.0)));
        let (g, u) = (rt.alloc_tensor_f32(&[t, INTER])?, rt.alloc_tensor_f32(&[t, INTER])?);
        let gu = rt.alloc_tensor_f32(&[t, 2 * INTER])?;
        let run = |lane: Lane| -> Res<()> {
            match lane {
                Lane::Prod => {
                    gemm(&x, &gate_w, &g, BACKEND)?;
                    gemm(&x, &up_w, &u, BACKEND)
                }
                Lane::Cand => gemm(&x, &packed_w, &gu, BACKEND),
            }
        };
        // Gate: the packed GEMM's two halves are the separate GEMMs' outputs.
        rt.set_async_encode(false)?;
        run(Lane::Prod)?;
        run(Lane::Cand)?;
        rt.synchronize()?;
        let (gv, uv, guv) = (g.buffer.read_f32(), u.buffer.read_f32(), gu.buffer.read_f32());
        let mut worst = 0.0f32;
        for r in 0..t {
            for c in 0..INTER {
                worst = worst
                    .max((gv[r * INTER + c] - guv[r * 2 * INTER + c]).abs())
                    .max((uv[r * INTER + c] - guv[r * 2 * INTER + INTER + c]).abs());
            }
        }
        finite_nonzero(&format!("T={t} packed gate|up"), &gu, t * 2 * INTER)?;
        println!("T={t}: gate passed; packed vs separate max |diff| = {worst:.3e}");
        for (mode, asynchronous, reps) in [("async", true, 16), ("sync", false, 4)] {
            rt.set_async_encode(asynchronous)?;
            paired_lanes(rt, t, &format!("gate_up_{mode}"), GATE_UP_LANES, reps, ROUNDS, run)?;
        }
        rt.set_async_encode(false)?;
    }
    Ok(())
}

// ------------------------------------------------------------- the model ---

use std::path::{Path, PathBuf};

use tessl::infer_trace;
use tessl::qwen35_model::{LayerKind, LayerPart, LogitRows, Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;

/// Where the model's weights come from.
#[derive(Clone, Debug)]
enum Weights {
    /// `Qwen35Model::random` at the 2B's shapes, seeded.
    Random,
    /// A checkpoint directory, already checked: its `config.json` and its one
    /// `*.safetensors` file.
    Dir { dir: PathBuf, st: PathBuf },
}

/// What the run was asked to time, from the command line.
struct Opts {
    weights: Weights,
    scan: qwen35::GdnScanSlice,
    /// Samples per timing (after `WARMUP` warm-ups).
    iters: usize,
}

/// The one `*.safetensors` file in `dir`; none or several is an error.
fn checkpoint_file(dir: &Path) -> Res<PathBuf> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("--weights {}: {e}", dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let path = entry.map_err(|e| format!("--weights {}: {e}", dir.display()))?.path();
        if path.is_file() && path.extension().is_some_and(|x| x == "safetensors") {
            files.push(path);
        }
    }
    match files.len() {
        1 => Ok(files.remove(0)),
        n => Err(format!(
            "--weights {}: expected exactly one .safetensors file, found {n}",
            dir.display()
        )),
    }
}

/// Check `--weights=DIR` before any GPU work: a directory holding a
/// `config.json` and exactly one `*.safetensors` file.
fn checkpoint(dir: &Path) -> Res<Weights> {
    let cfg = dir.join("config.json");
    if !cfg.is_file() {
        return Err(format!("--weights {}: no config.json", dir.display()));
    }
    let st = checkpoint_file(dir)?;
    // Parse both now, on the CPU: a config or checkpoint that cannot load is
    // refused here, not after GPU init.
    Qwen35Config::from_config_file(&cfg).map_err(|e| format!("--weights {}: {e}", dir.display()))?;
    SafeTensors::open(&st).map_err(|e| format!("--weights {}: {e}", st.display()))?;
    Ok(Weights::Dir {
        dir: dir.to_path_buf(),
        st,
    })
}

/// The model to time: bf16 (the production precision) with the packed head.
fn build_model(rt: &Arc<GpuRuntime>, o: &Opts) -> Res<Qwen35Model> {
    let t0 = Instant::now();
    let model = match &o.weights {
        Weights::Random => Qwen35Model::random(rt, Qwen35Config::qwen35_2b()?, Precision::Bf16, 0x51)?,
        Weights::Dir { dir, st } => {
            let cfg = Qwen35Config::from_config_file(&dir.join("config.json"))?;
            let st = SafeTensors::open(st)?;
            Qwen35Model::load(rt, &st, "model.language_model.", cfg, Precision::Bf16)?
        }
    };
    println!(
        "weights ready in {:.1} s: {}",
        t0.elapsed().as_secs_f64(),
        model.describe()
    );
    Ok(model.with_gdn_scan_slice(o.scan))
}

/// Deterministic token ids below `vocab`.
fn token_ids(n: usize, vocab: u32) -> Vec<u32> {
    // `n` is bounded by `MAX_TOKENS` at argument parsing; a larger count would
    // silently wrap to a short sequence, so it is a bug here, not a truncation.
    let n = u32::try_from(n).expect("token count exceeds u32; refused at argument parsing");
    (0..n).map(|i| i.wrapping_mul(7919).wrapping_add(13) % vocab).collect()
}

fn min_of(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Time `f` (which does its own waiting) `WARMUP` times untimed and `iters`
/// times timed; milliseconds per sample, in run order.
fn samples_ms(iters: usize, mut f: impl FnMut() -> Res<()>) -> Res<Vec<f64>> {
    for _ in 0..WARMUP {
        f()?;
    }
    let mut out = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f()?;
        out.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(out)
}

/// A timing's min and median, ms.
#[derive(Clone, Copy)]
struct Stat {
    min: f64,
    median: f64,
}

impl Stat {
    fn of(samples: Vec<f64>) -> Res<Self> {
        Ok(Self {
            min: min_of(&samples),
            median: common::median(samples)?,
        })
    }

    fn tok_s(&self, tokens: usize) -> f64 {
        tokens as f64 / self.min * 1e3
    }
}

/// `v` finite and not all zero: a forward that wrote nothing would otherwise
/// post the best number in the table.
fn finite_nonzero_slice(what: &str, v: &[f32]) -> Res<()> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(format!("{what}: element {i} of {} is {}", v.len(), v[i]));
    }
    if v.iter().all(|&x| x == 0.0) {
        return Err(format!("{what}: all {} elements are zero", v.len()));
    }
    Ok(())
}

/// The final norm of a prefill of `ids` through every layer, read back.
fn tower_final_norm(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ids: &[u32]) -> Res<Tensor> {
    let layers = model.config().layers.len();
    let out = rt.alloc_tensor_f32(&[ids.len(), model.config().hidden as usize])?;
    let mut s = model.begin(ids)?;
    s.advance_to(layers)?;
    s.final_norm_f32(&out)?;
    rt.synchronize()?;
    Ok(out)
}

/// The gate: one untimed forward whose final norm and last-row logits are
/// finite and not all zero.
fn gate(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ids: &[u32]) -> Res<()> {
    let t = ids.len();
    let norm = tower_final_norm(rt, model, ids)?;
    finite_nonzero("final norm", &norm, t * model.config().hidden as usize)?;
    let logits = model.forward_rows(ids, LogitRows::Last, false)?.logits;
    finite_nonzero_slice("last-row logits", &logits)
}

/// The tower lane: `begin` + `advance_to` + `final_norm_f32`, begin included.
fn time_tower(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ids: &[u32], iters: usize) -> Res<Stat> {
    let layers = model.config().layers.len();
    let out = rt.alloc_tensor_f32(&[ids.len(), model.config().hidden as usize])?;
    Stat::of(samples_ms(iters, || {
        let mut s = model.begin(ids)?;
        s.advance_to(layers)?;
        s.final_norm_f32(&out)?;
        rt.synchronize()
    })?)
}

/// All layers and the final norm on a session that is already begun: the
/// earlier bench's "forward" (preallocated activations, no embedding).
fn time_layers(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ids: &[u32], iters: usize) -> Res<Stat> {
    let layers = model.config().layers.len();
    let out = rt.alloc_tensor_f32(&[ids.len(), model.config().hidden as usize])?;
    for _ in 0..WARMUP {
        let mut s = model.begin(ids)?;
        s.advance_to(layers)?;
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut s = model.begin(ids)?;
        rt.synchronize()?;
        let t0 = Instant::now();
        s.advance_to(layers)?;
        s.final_norm_f32(&out)?;
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    Stat::of(samples)
}

/// `Qwen35Model::forward_rows(Last)`: the tower plus the head's one row.
fn time_forward_last(model: &Qwen35Model, ids: &[u32], iters: usize) -> Res<Stat> {
    Stat::of(samples_ms(iters, || {
        model.forward_rows(ids, LogitRows::Last, false).map(drop)
    })?)
}

/// `Qwen35Model::prefill` to the first logits, keeping the decode state.
fn time_prefill(model: &Qwen35Model, ids: &[u32], max_new: u32, iters: usize) -> Res<Stat> {
    Stat::of(samples_ms(iters, || model.prefill(ids, max_new).map(drop))?)
}

/// One row of the per-part table.
struct PartRow {
    group: &'static str,
    name: String,
    ms: f64,
}

/// What `time_parts` measured at one T.
struct Parts {
    gdn_layer: f64,
    attn_layer: f64,
    mlp: f64,
    rows: Vec<PartRow>,
}

fn part_name(part: LayerPart) -> &'static str {
    match part {
        LayerPart::InputNorm => "input norm",
        LayerPart::Mixer => "mixer",
        LayerPart::PostNorm => "post norm",
        LayerPart::Mlp => "MLP (gate, up, swiglu, down)",
    }
}

/// Per-layer and per-part times through `Staged::encode_part`, `REPS` per
/// command buffer. Repeats move the residual stream (the mixer and MLP add to
/// it in place), so the residual is read back after and must still be finite.
fn time_parts(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ids: &[u32], iters: usize) -> Res<Parts> {
    let kinds = &model.config().layers;
    let find = |k: LayerKind| {
        kinds
            .iter()
            .position(|&x| x == k)
            .ok_or_else(|| format!("the model has no {k:?} layer to time"))
    };
    let (gdn, attn) = (find(LayerKind::LinearAttention)?, find(LayerKind::FullAttention)?);
    let s = model.begin(ids)?;
    let run = |layer: usize, parts: &[LayerPart]| -> Res<f64> {
        time_ms(rt, REPS, iters, || {
            parts.iter().try_for_each(|&p| s.encode_part(layer, p))
        })
    };
    let mixer = [LayerPart::InputNorm, LayerPart::Mixer];
    let mlp_parts = [LayerPart::PostNorm, LayerPart::Mlp];
    let gdn_layer = run(gdn, &mixer)?;
    let attn_layer = run(attn, &mixer)?;
    let mlp = run(gdn, &mlp_parts)?;
    let mut rows = Vec::new();
    for (group, layer) in [("gdn", gdn), ("attn", attn)] {
        for part in [LayerPart::InputNorm, LayerPart::Mixer] {
            rows.push(PartRow {
                group,
                name: part_name(part).to_string(),
                ms: run(layer, &[part])?,
            });
        }
    }
    for part in mlp_parts {
        rows.push(PartRow {
            group: "mlp",
            name: part_name(part).to_string(),
            ms: run(gdn, &[part])?,
        });
    }
    let out = rt.alloc_tensor_f32(&[ids.len(), model.config().hidden as usize])?;
    s.final_norm_f32(&out)?;
    rt.synchronize()?;
    finite_nonzero(
        "final norm after the part loops",
        &out,
        ids.len() * model.config().hidden as usize,
    )?;
    Ok(Parts {
        gdn_layer,
        attn_layer,
        mlp,
        rows,
    })
}

/// Dispatches one tower forward encodes (not timed).
fn dispatch_count(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ids: &[u32]) -> Res<usize> {
    rt.synchronize()?;
    rt.take_dispatch_count();
    tower_final_norm(rt, model, ids)?;
    Ok(rt.take_dispatch_count())
}

/// The default run: gate, then the lanes above at each T.
fn run_model(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ts: &[usize], o: &Opts, check_only: bool) -> Res<()> {
    let vocab = model.config().vocab;
    rt.set_async_encode(true)?;
    for &t in ts {
        gate(rt, model, &token_ids(t, vocab)).map_err(|e| format!("T={t}: gate failed: {e}"))?;
        println!("T={t}: gate passed (final norm and last-row logits finite and not all zero)");
    }
    if check_only {
        rt.set_async_encode(false)?;
        println!("--check-only: nothing timed");
        return Ok(());
    }
    let kinds = &model.config().layers;
    let n_attn = kinds.iter().filter(|&&k| k == LayerKind::FullAttention).count();
    let n_gdn = kinds.len() - n_attn;
    struct Row {
        t: usize,
        tower: Stat,
        layers: Stat,
        fwd: Stat,
        prefill: Stat,
        parts: Parts,
        dispatches: usize,
    }
    let mut rows = Vec::new();
    for &t in ts {
        let ids = token_ids(t, vocab);
        let dispatches = dispatch_count(rt, model, &ids)?;
        let tower = time_tower(rt, model, &ids, o.iters)?;
        println!("METRIC model_forward_ms_t{t} {:.3}", tower.median);
        let layers = time_layers(rt, model, &ids, o.iters)?;
        let fwd = time_forward_last(model, &ids, o.iters)?;
        let prefill = time_prefill(model, &ids, 1, o.iters)?;
        let parts = time_parts(rt, model, &ids, o.iters)?;
        rows.push(Row {
            t,
            tower,
            layers,
            fwd,
            prefill,
            parts,
            dispatches,
        });
    }
    rt.set_async_encode(false)?;

    println!();
    println!(
        "Qwen35Model prefill (ms, min / median of {}), async encode. tower = begin + layers + final norm; \
         layers = layers + final norm only; forward_rows = tower + head (last row); prefill = forward_rows + decode state",
        o.iters
    );
    println!(
        "{:>6} | {:>9} {:>9} {:>9} | {:>9} {:>9} | {:>9} | {:>9} | {:>10} {:>10} {:>10} | {:>9}",
        "T",
        "tower min",
        "median",
        "layers",
        "fwd min",
        "median",
        "prefill",
        "median",
        "tower tok/s",
        "fwd tok/s",
        "pre tok/s",
        "launches"
    );
    for r in &rows {
        println!(
            "{:>6} | {:>9.2} {:>9.2} {:>9.2} | {:>9.2} {:>9.2} | {:>9.2} | {:>9.2} | {:>10.0} {:>10.0} {:>10.0} | {:>9}",
            r.t,
            r.tower.min,
            r.tower.median,
            r.layers.min,
            r.fwd.min,
            r.fwd.median,
            r.prefill.min,
            r.prefill.median,
            r.tower.tok_s(r.t),
            r.fwd.tok_s(r.t),
            r.prefill.tok_s(r.t),
            r.dispatches,
        );
    }

    println!();
    println!(
        "per layer (ms, median of {}, {REPS} per command buffer, Staged::encode_part). GDN / attention layer = \
         input norm + mixer; MLP = post-norm + MLP",
        o.iters
    );
    println!(
        "{:>6} {:>9} {:>9} {:>9} | {:>11} {:>11}",
        "T", "GDN", "attn", "MLP", "sum layers", "layers min"
    );
    for r in &rows {
        let p = &r.parts;
        let summed = n_gdn as f64 * p.gdn_layer + n_attn as f64 * p.attn_layer + kinds.len() as f64 * p.mlp;
        println!(
            "{:>6} {:>9.3} {:>9.3} {:>9.3} | {:>11.2} {:>11.2}",
            r.t, p.gdn_layer, p.attn_layer, p.mlp, summed, r.layers.min
        );
    }
    println!();
    println!("per part (ms)");
    print!("{:<6} {:<30}", "", "part");
    for r in &rows {
        print!(" {:>9}", format!("T={}", r.t));
    }
    println!();
    for i in 0..rows[0].parts.rows.len() {
        let head = &rows[0].parts.rows[i];
        print!("{:<6} {:<30}", head.group, head.name);
        for r in &rows {
            print!(" {:>9.3}", r.parts.rows[i].ms);
        }
        println!();
    }
    Ok(())
}

/// `--model-forward`: the tower lane alone, METRIC lines.
fn run_model_forward(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ts: &[usize], o: &Opts) -> Res<()> {
    rt.set_async_encode(true)?;
    for &t in ts {
        let ids = token_ids(t, model.config().vocab);
        gate(rt, model, &ids).map_err(|e| format!("T={t}: gate failed: {e}"))?;
        let tower = time_tower(rt, model, &ids, o.iters)?;
        println!("T={t}: forward median {:.3} ms, min {:.3} ms", tower.median, tower.min);
        println!("METRIC model_forward_ms_t{t} {:.3}", tower.median);
    }
    rt.set_async_encode(false)?;
    Ok(())
}

/// Decode steps timed per run, after `DECODE_WARMUP`; and steps counted
/// with `infer_trace` on, kept apart so tracing is not in the timed ones.
const DECODE_STEPS: usize = 32;
const DECODE_WARMUP: usize = 2;
const DECODE_TRACED: usize = 4;

/// `--decode[=P]`: `Qwen35Model::prefill` of `P` tokens, then `Decode::step`
/// per token.
fn run_decode(rt: &Arc<GpuRuntime>, model: &Qwen35Model, prefix_len: usize, o: &Opts) -> Res<()> {
    let vocab = model.config().vocab;
    let ids = token_ids(prefix_len, vocab);
    let max_new = (DECODE_WARMUP + DECODE_STEPS + DECODE_TRACED) as u32;
    rt.set_async_encode(true)?;
    gate(rt, model, &ids).map_err(|e| format!("P={prefix_len}: gate failed: {e}"))?;
    let prefill = time_prefill(model, &ids, max_new, o.iters)?;
    println!(
        "prefill P={prefix_len}: min {:.2} ms, median {:.2} ms ({:.0} tok/s at min)",
        prefill.min,
        prefill.median,
        prefill.tok_s(prefix_len)
    );
    let (mut dec, first) = model.prefill(&ids, max_new)?;
    finite_nonzero_slice("prefill's first logits", &first)?;
    let pattern = token_ids(DECODE_WARMUP + DECODE_STEPS + DECODE_TRACED, vocab);
    let mut next = pattern.iter().copied();
    let mut step = |dec: &mut tessl::qwen35_model::Decode<'_>| -> Res<f64> {
        let id = next.next().ok_or("decode: token pattern exhausted")?;
        let t0 = Instant::now();
        let logits = dec.step(id)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        finite_nonzero_slice("decoded logits", &logits)?;
        Ok(ms)
    };
    for _ in 0..DECODE_WARMUP {
        step(&mut dec)?;
    }
    let mut samples = Vec::with_capacity(DECODE_STEPS);
    for _ in 0..DECODE_STEPS {
        samples.push(step(&mut dec)?);
    }
    let stat = Stat::of(samples)?;
    // The same steps with the runtime's counters on: per token, what a
    // GPU-resident decode loop would remove.
    infer_trace::set_enabled(true);
    let mut counts = Vec::with_capacity(DECODE_TRACED);
    for _ in 0..DECODE_TRACED {
        let before = infer_trace::snapshot();
        step(&mut dec)?;
        counts.push(infer_trace::snapshot().since(&before));
    }
    infer_trace::set_enabled(false);
    rt.set_async_encode(false)?;

    let per_token = |f: fn(&infer_trace::Snapshot) -> u64| -> (u64, u64) {
        let v: Vec<u64> = counts.iter().map(f).collect();
        (
            v.iter().copied().min().unwrap_or(0),
            v.iter().copied().max().unwrap_or(0),
        )
    };
    let (allocs, waits, commits, dispatches) = (
        per_token(|s| s.cold_allocs),
        per_token(|s| s.sync_waits),
        per_token(|s| s.commits),
        per_token(|s| s.dispatches),
    );
    let wait_us = counts.iter().map(|s| s.sync_wait_us).min().unwrap_or(0);
    let range = |(lo, hi): (u64, u64)| {
        if lo == hi {
            format!("{lo}")
        } else {
            format!("{lo}..{hi}")
        }
    };
    println!(
        "decode after P={prefix_len}: {DECODE_STEPS} steps min {:.3} ms, median {:.3} ms -> {:.1} tok/s at min, {:.1} at median",
        stat.min,
        stat.median,
        1e3 / stat.min,
        1e3 / stat.median
    );
    println!(
        "Decode::step per token ({DECODE_TRACED} traced steps): cold allocs {}, host waits {}, commits {}, \
         dispatches {}, wait {wait_us} us (min)",
        range(allocs),
        range(waits),
        range(commits),
        range(dispatches)
    );
    println!("METRIC model_prefill_ms_p{prefix_len} {:.3}", prefill.min);
    println!("METRIC model_decode_ms_min {:.3}", stat.min);
    println!("METRIC model_decode_ms_median {:.3}", stat.median);
    println!("METRIC model_decode_toks_per_s {:.2}", 1e3 / stat.min);
    println!("METRIC model_decode_cold_allocs_per_token {}", range(allocs));
    println!("METRIC model_decode_sync_waits_per_token {}", range(waits));
    println!("METRIC model_decode_commits_per_token {}", range(commits));
    println!("METRIC model_decode_dispatches_per_token {}", range(dispatches));
    println!("METRIC model_decode_sync_wait_us {wait_us}");
    Ok(())
}

/// Power source and mode, for the record next to a timing: `pmset` is the
/// platform's own report, shown as it prints it.
fn print_power_state() {
    for args in [&["-g", "batt"][..], &["-g"][..]] {
        match std::process::Command::new("pmset").args(args).output() {
            Ok(o) if o.status.success() => {
                for line in String::from_utf8_lossy(&o.stdout).lines() {
                    let l = line.trim();
                    if args.len() == 2 && (l.contains("Now drawing") || l.contains("%")) || l.starts_with("powermode") {
                        println!("power: {l}");
                    }
                }
            }
            Ok(o) => println!("power: pmset {args:?} exited {}", o.status),
            Err(e) => println!("power: pmset {args:?} could not run: {e}"),
        }
    }
}

// --------------------------------------------------------------------- main ---

/// Largest token count (T, or `--decode=P`) accepted. Well above any real
/// context, and below `u32::MAX` so every position fits the kernels' `u32`
/// sequence arguments and `token_ids` cannot wrap.
const MAX_TOKENS: usize = 1 << 24;

/// Refuse a token count above [`MAX_TOKENS`] before any GPU work.
fn bounded_tokens(n: usize, what: &str) -> Result<usize, String> {
    if n > MAX_TOKENS {
        return Err(format!("{what} must be at most {MAX_TOKENS}, got {n}"));
    }
    Ok(n)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut check_only = false;
    let mut paired_attn = false;
    let mut attn_tiles = false;
    let mut decode: Option<usize> = None;
    let mut paired_rec = false;
    let mut paired_gate_up = false;
    let mut model_forward = false;
    let mut opts = Opts {
        weights: Weights::Random,
        scan: qwen35::GdnScanSlice::default(),
        iters: ITERS,
    };
    let mut ts = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--paired-gdn-recurrent" {
            paired_rec = true;
        } else if arg == "--paired-gate-up" {
            paired_gate_up = true;
        } else if arg == "--model-forward" {
            model_forward = true;
        } else if arg == "--decode" {
            decode = Some(4096);
        } else if let Some(p) = arg.strip_prefix("--decode=") {
            let p: usize = p
                .parse()
                .map_err(|_| format!("--decode=P wants a prefix length, got {p:?}"))?;
            if p == 0 {
                return Err("--decode prefix must be positive".into());
            }
            decode = Some(bounded_tokens(p, "--decode prefix")?);
        } else if arg == "--check-only" {
            check_only = true;
        } else if arg == "--paired-attn" {
            paired_attn = true;
        } else if arg == "--attn-tiles" {
            attn_tiles = true;
        } else if arg == "--gdn-scan32" {
            opts.scan = qwen35::GdnScanSlice::Cols32;
        } else if let Some(n) = arg.strip_prefix("--iters=") {
            opts.iters = n
                .parse()
                .ok()
                .filter(|n| (1..=MAX_ITERS).contains(n))
                .ok_or_else(|| format!("--iters=N wants a count in 1..={MAX_ITERS}, got {n:?}"))?;
        } else if let Some(dir) = arg.strip_prefix("--weights=") {
            if dir.is_empty() {
                return Err("--weights=DIR wants a checkpoint directory".into());
            }
            opts.weights = checkpoint(Path::new(dir))?;
        } else {
            let t: usize = arg.parse().map_err(|_| {
                format!(
                    "expected a token count, --check-only, --paired-attn, --attn-tiles, --paired-gdn-recurrent, --paired-gate-up, \
                     --model-forward, --weights=DIR, --gdn-scan32, --iters=N or --decode[=P], got {arg:?}"
                )
            })?;
            if t == 0 {
                return Err("T must be positive".into());
            }
            ts.push(bounded_tokens(t, "T")?);
        }
    }
    if attn_tiles && (check_only || paired_attn || paired_rec || paired_gate_up || model_forward || decode.is_some()) {
        return Err("--attn-tiles takes token counts only".into());
    }
    if paired_attn && check_only {
        return Err("--paired-attn times both kernels; it does not combine with --check-only".into());
    }
    if decode.is_some()
        && (check_only || paired_attn || paired_rec || paired_gate_up || model_forward || !ts.is_empty())
    {
        return Err("--decode times decode only; it takes no T, --check-only, --model-forward or --paired-*".into());
    }
    if paired_rec && (check_only || paired_attn) {
        return Err("--paired-gdn-recurrent takes batch sizes only".into());
    }
    let modes = [paired_gate_up, model_forward].iter().filter(|&&m| m).count();
    if modes > 1 || (modes == 1 && (check_only || paired_attn || paired_rec || decode.is_some())) {
        return Err("--paired-gate-up and --model-forward each take token counts only".into());
    }
    let paired = paired_attn || attn_tiles || paired_rec || paired_gate_up;
    if paired && (matches!(opts.weights, Weights::Dir { .. }) || opts.scan != qwen35::GdnScanSlice::default()) {
        return Err("--weights and --gdn-scan32 apply to the model lanes, not --paired-*".into());
    }
    if paired_gate_up {
        let rt = GpuRuntime::new()?;
        println!("device: {}", rt.device_name());
        run_paired_gate_up(&rt, if ts.is_empty() { &[512, 2048] } else { &ts })?;
        return Ok(());
    }
    if paired_rec {
        let rt = GpuRuntime::new()?;
        println!("device: {}", rt.device_name());
        rt.set_async_encode(true)?;
        run_paired_gdn_recurrent(&rt, if ts.is_empty() { &[1] } else { &ts })?;
        rt.set_async_encode(false)?;
        return Ok(());
    }
    let rt = GpuRuntime::new()?;
    if !rt.has_tensorops() {
        return Err("bf16 GEMMs need the TensorOps backend, which this device lacks".into());
    }
    println!("device: {}", rt.device_name());
    if attn_tiles {
        rt.set_async_encode(true)?;
        run_attn_tiles(&rt, if ts.is_empty() { &[2048] } else { &ts })?;
        rt.set_async_encode(false)?;
        return Ok(());
    }
    if paired_attn {
        rt.set_async_encode(true)?;
        run_paired_attn(&rt, if ts.is_empty() { &[200, 8192] } else { &ts })?;
        rt.set_async_encode(false)?;
        return Ok(());
    }
    print_power_state();
    println!("gdn scan: {:?}", opts.scan);
    println!(
        "weights: {}",
        match &opts.weights {
            Weights::Random => "random (seed 0x51)".to_string(),
            Weights::Dir { dir, .. } => dir.display().to_string(),
        }
    );
    let model = build_model(&rt, &opts)?;
    if let Some(prefix_len) = decode {
        return run_decode(&rt, &model, prefix_len, &opts).map_err(Into::into);
    }
    if model_forward {
        let ts = if ts.is_empty() { vec![1024, 2048] } else { ts };
        return run_model_forward(&rt, &model, &ts, &opts).map_err(Into::into);
    }
    let ts = if ts.is_empty() { DEFAULT_T.to_vec() } else { ts };
    run_model(&rt, &model, &ts, &opts, check_only).map_err(Into::into)
}
