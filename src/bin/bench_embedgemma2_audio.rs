//! End-to-end latency of `tessl::embedgemma2_audio` on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/embedgemma2_mm_ref \
//!   cargo run --release --bin bench_embedgemma2_audio
//! ```
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,` (default: the first
//! three): `clip_1.0s` and `clip_3.3s` (the golden clips of the reference
//! directory) and `batch_8x30s` (eight synthetic clips of 3000 frames; see
//! `common/audio_workloads.rs`); by name also `batch_4x5s`, `batch_8x10s` and
//! `clip_max` (one clip of `MAX_FRAMES_PER_FORWARD` frames). `BENCH_ITERS`
//! (default 10) and `BENCH_WARMUP` (default 3) are per workload.
//! `BENCH_PRECISION=f32|bf16` (default `f32`) is the tower's operand precision
//! ([`GemmOperands`]; `EmbedGemma2Audio::load_with`). The runtime queues
//! dispatches (`set_async_encode(true)`, which `BENCH_ASYNC=0` turns off) so
//! that every build of this binary, whichever way its `encode_audio` waits,
//! times the pipelined forward; each object carries `"async":0|1`.
//!
//! `--kernels` instead runs the six `eg2a_*` kernels through their public
//! functions on random operands, with no checkpoint (`run_audio_kernels`).
//!
//! What is timed: one `encode_audio` call, from host features to host soft
//! tokens: allocation, upload, the forwards, the wait and the read back. Not
//! timed: the model load and the log-mel front end (the caller's).
//!
//! Prints one JSON array on stdout, one object per workload: `{workload,
//! ms_min, ms_median, iters, async, precision, forwards, frames, soft_err_max,
//! soft_cos_min, device_peak_mib}`, the format `bench_paired` reads (`--format
//! workloads`). Before anything is timed the soft tokens must be finite and,
//! for the golden clips, within `1e-4` of their largest magnitude of
//! transformers' (the bound of `tests/embedgemma2_audio_model.rs`) under `f32`,
//! at cosine `0.9997` or better under `bf16`; `soft_err_max` is the observed
//! worst and `soft_cos_min` the worst cosine (`null` under `f32`, and for a
//! workload with no reference).
//!
//! The binary uses only the library's public `load_with` and `encode_audio`.
//! An older revision, which has `load` and no operand choice, is benchmarked
//! with this file's one `load_with(.., operands)` call replaced by `load(..)` (its
//! `precision` field is then not a measurement: run it at `f32`);
//! `docs/embedgemma2.md` (the audio tower bf16 and speed section) says how the
//! before lane was built.

mod common;

#[path = "common/audio_workloads.rs"]
mod audio_workloads;

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2_audio::{AudioConfig, EmbedGemma2Audio};
use tessl::gemm::GemmOperands;
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

use common::env_usize;

const MIB: f64 = (1u64 << 20) as f64;
/// The soft-token bound `tests/embedgemma2_audio_model.rs` states.
const SOFT_BOUND: f64 = 1e-4;
/// A bf16 forward's soft tokens against transformers' fp32 ones: the floor
/// `tests/embedgemma2_audio_model.rs` holds them to.
const SOFT_COS_FLOOR: f64 = 0.9997;

/// `--kernels`: the six `eg2a_*` kernels the tower's stages launch
/// (`eg2a_sub_conv0_ln_relu`, `eg2a_sub_conv1_ln_relu`, `eg2a_rms_norm`,
/// `eg2a_ew`, `eg2a_lconv`, `eg2a_attn`) through their public functions on
/// random operands over a two-clip table, with no checkpoint. Each is run once
/// into a NaN-poisoned output and checked before it is timed: `rms_norm`,
/// `ew` and `lconv` against an f64 host reference of the formula in their doc
/// comments; the two subsampling convs and the attention (whose references are
/// the model-level tests') for a finite, non-trivial output (non-negative for
/// the convs' ReLU). One `METRIC eg2a_{kernel}_ms` per kernel.
fn run_audio_kernels(warmup: usize, iters: usize) -> Result<(), String> {
    use common::{check_against, check_lane_output, fill_affine, host_rms, poisoned_f32, time_lane, upload_f32};
    use tessl::embedgemma2_audio::{
        attention, conv_out, ew, lconv, rms_norm, sub_conv0_ln_relu, sub_conv1_ln_relu, AttentionSpec, Clamp,
        ClipTable, Ew, Lconv, RmsNorm,
    };

    const EPS: f32 = 1e-6;
    let rt = GpuRuntime::new_inference()?;
    let as_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    let clamp = |x: f64, c: Clamp| x.max(f64::from(c.lo)).min(f64::from(c.hi));
    let silu = |x: f64| x / (1.0 + (-x).exp());
    let table = ClipTable::new(&rt, &[40, 23])?;
    let rows = table.rows;

    // The subsampling convs: feats -> conv0 (+ layer norm, ReLU) -> conv1.
    {
        let (mels, c0, c1) = (16usize, 8usize, 4usize);
        let mels1 = conv_out(mels);
        let mels2 = conv_out(mels1);
        let feats = upload_f32(&rt, &fill_affine(table.frames * mels, 51, 1.5, 0.0))?;
        let fmask = rt.alloc_buffer_from_u32(&vec![1u32; table.frames])?;
        let (w0, g0) = (
            upload_f32(&rt, &fill_affine(c0 * 9, 52, 0.5, 0.0))?,
            upload_f32(&rt, &fill_affine(c0, 53, 0.3, 1.0))?,
        );
        let (w1, g1) = (
            upload_f32(&rt, &fill_affine(9 * c0 * c1, 54, 0.5, 0.0))?,
            upload_f32(&rt, &fill_affine(c1, 55, 0.3, 1.0))?,
        );
        let mid = poisoned_f32(&rt, table.mid_rows * mels1 * c0)?;
        let out = poisoned_f32(&rt, rows * mels2 * c1)?;
        let conv0 = || sub_conv0_ln_relu(&rt, &feats, &fmask, &w0, &g0, &mid, &table, mels as u32, c0 as u32, EPS);
        let conv1 = || {
            sub_conv1_ln_relu(
                &rt,
                &mid,
                &fmask,
                &w1,
                &g1,
                &out,
                &table,
                mels1 as u32,
                c0 as u32,
                c1 as u32,
                EPS,
            )
        };
        conv0()?;
        conv1()?;
        rt.synchronize()?;
        for (lane, got) in [
            (
                "eg2a_sub_conv0_ln_relu",
                mid.read_f32()[..table.mid_rows * mels1 * c0].to_vec(),
            ),
            ("eg2a_sub_conv1_ln_relu", out.read_f32()[..rows * mels2 * c1].to_vec()),
        ] {
            check_lane_output(lane, &got)?;
            if let Some(i) = got.iter().position(|&x| x < 0.0) {
                return Err(format!(
                    "{lane}: element {i} is {}; a ReLU output is never negative",
                    got[i]
                ));
            }
        }
        time_lane(&rt, "eg2a_sub_conv0_ln_relu", warmup, iters, conv0)?;
        time_lane(&rt, "eg2a_sub_conv1_ln_relu", warmup, iters, conv1)?;
    }

    // The row norm: `rms_norm(x) * weight`, no clamps.
    {
        let (nrows, dim) = (5usize, 32usize);
        let lane = "eg2a_rms_norm";
        let x_h = fill_affine(nrows * dim, 56, 3.0, 0.0);
        let w_h = fill_affine(dim, 57, 0.4, 1.0);
        let (x, w, out) = (
            upload_f32(&rt, &x_h)?,
            upload_f32(&rt, &w_h)?,
            poisoned_f32(&rt, nrows * dim)?,
        );
        let spec = RmsNorm::new(EPS, Some(&w));
        let dispatch = || rms_norm(&rt, &x, &out, nrows as u32, dim as u32, &spec);
        dispatch()?;
        rt.synchronize()?;
        let want: Vec<f64> = (0..nrows)
            .flat_map(|r| host_rms(&as_f64(&x_h[r * dim..(r + 1) * dim]), Some(&w_h), f64::from(EPS)))
            .collect();
        let got = out.read_f32();
        check_lane_output(lane, &got[..nrows * dim])?;
        check_against(lane, &got[..nrows * dim], &want, 1e-5, 2e-4)?;
        time_lane(&rt, lane, warmup, iters, dispatch)?;
    }

    // The elementwise stage: `clamp(post)(silu(clamp(pre)(x)))`.
    {
        let n = 200usize;
        let lane = "eg2a_ew";
        let x_h = fill_affine(n, 58, 4.0, 0.0);
        let (x, out) = (upload_f32(&rt, &x_h)?, poisoned_f32(&rt, n)?);
        let spec = Ew {
            pre: Clamp::new(-1.5, 2.5)?,
            silu: true,
            post: Clamp::new(-0.25, 1.75)?,
        };
        let dispatch = || ew(&rt, &x, &out, n as u32, &spec);
        dispatch()?;
        rt.synchronize()?;
        let want: Vec<f64> = x_h
            .iter()
            .map(|&v| clamp(silu(clamp(f64::from(v), spec.pre)), spec.post))
            .collect();
        let got = out.read_f32();
        check_lane_output(lane, &got[..n])?;
        check_against(lane, &got[..n], &want, 1e-5, 1e-4)?;
        time_lane(&rt, lane, warmup, iters, dispatch)?;
    }

    // The light conv's middle: GLU, depthwise causal conv per clip, norm, SiLU.
    {
        let (hidden, taps) = (24usize, 5usize);
        let lane = "eg2a_lconv";
        let x_h = fill_affine(rows * 2 * hidden, 59, 3.0, 0.0);
        let w_h = fill_affine(hidden * taps, 60, 0.8, 0.0);
        let nw_h = fill_affine(hidden, 61, 0.4, 1.0);
        let (x, w, nw, out) = (
            upload_f32(&rt, &x_h)?,
            upload_f32(&rt, &w_h)?,
            upload_f32(&rt, &nw_h)?,
            poisoned_f32(&rt, rows * hidden)?,
        );
        let spec = Lconv {
            eps: EPS,
            norm_weight: &nw,
            glu: Clamp::new(-2.0, 2.0)?,
            pre: Clamp::new(-3.0, 3.0)?,
            post: Clamp::new(-1.5, 1.5)?,
        };
        let dispatch = || lconv(&rt, &x, &w, &out, &table, hidden as u32, taps as u32, &spec);
        dispatch()?;
        rt.synchronize()?;
        let glu = |s: usize, c: usize| -> f64 {
            let a = clamp(f64::from(x_h[s * 2 * hidden + c]), spec.glu);
            let g = clamp(f64::from(x_h[s * 2 * hidden + hidden + c]), spec.glu);
            a / (1.0 + (-g).exp())
        };
        let mut want = Vec::with_capacity(rows * hidden);
        for clip in &table.clips {
            for t in 0..clip.rows {
                let conv: Vec<f64> = (0..hidden)
                    .map(|c| {
                        (0..taps)
                            .filter_map(|j| {
                                // Rows before the clip's first read 0.
                                (t + j + 1)
                                    .checked_sub(taps)
                                    .map(|s| f64::from(w_h[c * taps + j]) * glu(clip.row0 + s, c))
                            })
                            .sum::<f64>()
                    })
                    .map(|v| clamp(v, spec.pre))
                    .collect();
                want.extend(
                    host_rms(&conv, Some(&nw_h), f64::from(EPS))
                        .into_iter()
                        .map(|v| clamp(silu(v), spec.post)),
                );
            }
        }
        let got = out.read_f32();
        check_lane_output(lane, &got[..rows * hidden])?;
        check_against(lane, &got[..rows * hidden], &want, 1e-4, 2e-4)?;
        time_lane(&rt, lane, warmup, iters, dispatch)?;
    }

    // The chunked relative-position attention, q, k and v in buffers of their own.
    {
        let (heads, head_dim, chunk, left) = (3usize, 10usize, 4usize, 5usize);
        let hidden = heads * head_dim;
        let past = left - 1;
        let ctx = chunk + past;
        let rel_rows = ctx / 2 + 1;
        let lane = "eg2a_attn";
        let (q, k, v) = (
            upload_f32(&rt, &fill_affine(rows * hidden, 62, 2.0, 0.0))?,
            upload_f32(&rt, &fill_affine(rows * hidden, 63, 2.0, 0.0))?,
            upload_f32(&rt, &fill_affine(rows * hidden, 64, 2.0, 0.0))?,
        );
        let rel = upload_f32(&rt, &fill_affine(rel_rows * hidden, 65, 0.5, 0.0))?;
        let q_mult = upload_f32(&rt, &fill_affine(head_dim, 66, 0.3, 0.6))?;
        let omask = rt.alloc_buffer_from_u32(&vec![1u32; rows])?;
        let out = poisoned_f32(&rt, rows * hidden)?;
        let spec = AttentionSpec {
            heads: heads as u32,
            head_dim: head_dim as u32,
            chunk: chunk as u32,
            past: past as u32,
            k_scale: ((1.0f64 + 1f64.exp()).ln() / 2f64.ln()) as f32,
            cap: 3.0,
            invalid: -1e9,
            q: Clamp::new(-1.2, 1.5)?,
            k: Clamp::new(-1.4, 1.1)?,
            v: Clamp::new(-1.3, 1.6)?,
            out: Clamp::new(-0.4, 0.5)?,
            stride: hidden as u32,
            q_off: 0,
            k_off: 0,
            v_off: 0,
        };
        let dispatch = || attention(&rt, &q, &k, &v, &rel, &q_mult, &omask, &out, &table, &spec);
        dispatch()?;
        rt.synchronize()?;
        let got = out.read_f32();
        check_lane_output(lane, &got[..rows * hidden])?;
        // The output clamp bounds every element.
        if let Some(i) = got[..rows * hidden]
            .iter()
            .position(|&x| x < spec.out.lo || x > spec.out.hi)
        {
            return Err(format!("{lane}: element {i} is {}, outside the output clamp", got[i]));
        }
        time_lane(&rt, lane, warmup, iters, dispatch)?;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    if std::env::args().nth(1).as_deref() == Some("--kernels") {
        if let Some(extra) = std::env::args().nth(2) {
            return Err(format!("unexpected argument {extra:?} after --kernels"));
        }
        let iters = env_usize("BENCH_ITERS", 10, 1)?;
        let warmup = env_usize("BENCH_WARMUP", 3, 0)?;
        let outcome = common::mark_incomplete_on_error(run_audio_kernels(warmup, iters));
        common::emit_kernel_trace();
        return outcome;
    }
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let workloads = audio_workloads::selected("BENCH_WORKLOADS")?;
    let operands = match std::env::var("BENCH_PRECISION").as_deref() {
        Err(_) | Ok("f32") => GemmOperands::ExactF32,
        Ok("bf16") => GemmOperands::Bf16,
        Ok(other) => return Err(format!("BENCH_PRECISION={other:?}: expected f32 or bf16")),
    };
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;
    let async_encode = match std::env::var("BENCH_ASYNC").as_deref() {
        Err(_) | Ok("1") => true,
        Ok("0") => false,
        Ok(other) => return Err(format!("BENCH_ASYNC={other:?}: expected 0 or 1")),
    };

    let cfg = AudioConfig::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(async_encode)?;
    let model = EmbedGemma2Audio::load_with(&rt, &st, "", cfg, operands)?;
    let loaded = rt.current_allocated_bytes();

    let mut rows = Vec::new();
    for w in &workloads {
        rt.reset_peak_allocated_bytes();
        let out = model.encode_audio(&w.clips, false)?;
        let (err, cos) = match operands {
            GemmOperands::ExactF32 => (
                audio_workloads::json_opt(audio_workloads::check(w, &out.soft_tokens, SOFT_BOUND)?),
                "null".to_string(),
            ),
            GemmOperands::Bf16 => {
                let seen = audio_workloads::check_cosine(w, &out.soft_tokens, SOFT_COS_FLOOR)?;
                (
                    audio_workloads::json_opt(seen.map(|(e, _)| e)),
                    audio_workloads::json_opt(seen.map(|(_, c)| c)),
                )
            }
        };
        for _ in 0..warmup {
            model.encode_audio(&w.clips, false)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.encode_audio(&w.clips, false)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        let device_peak = rt.peak_allocated_bytes().saturating_sub(loaded);
        rows.push(format!(
            "{{\"workload\":\"{}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"async\":{},\"precision\":\"{}\",\"forwards\":{},\"frames\":{},\"soft_err_max\":{err},\"soft_cos_min\":{cos},\"device_peak_mib\":{:.1}}}",
            w.name,
            ms[0],
            ms[ms.len() / 2],
            u8::from(async_encode),
            if operands == GemmOperands::Bf16 { "bf16" } else { "f32" },
            out.forwards,
            w.frames(),
            device_peak as f64 / MIB,
        ));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}
