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

/// `x` centred, scaled by `1 / sqrt(biased variance + eps)` and `gamma`, then
/// rectified: the subsampling convs' `LayerNorm(eps, weight, no bias)` + ReLU
/// over one position's channels, in f64.
fn host_ln_relu(x: &[f64], gamma: &[f32], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let inv = 1.0 / (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n + eps).sqrt();
    x.iter()
        .zip(gamma)
        .map(|(v, &g)| ((v - mean) * inv * f64::from(g)).max(0.0))
        .collect()
}

/// `--kernels`: the six `eg2a_*` kernels the tower's stages launch
/// (`eg2a_sub_conv0_ln_relu`, `eg2a_sub_conv1_ln_relu`, `eg2a_rms_norm`,
/// `eg2a_ew`, `eg2a_lconv`, `eg2a_attn`) through their public functions on
/// random operands over a two-clip table, with no checkpoint. Each is run once
/// into a NaN-poisoned output and checked before it is timed, all six against
/// an f64 host reference of the formula in their doc comments (the kernels'
/// own, written out from `kernels/embedgemma2_audio.metal`): `rms_norm`, `ew`
/// and `lconv` as before; the two subsampling convs over a frame mask with
/// holes (conv1 fed conv0's read-back output, so the two are checked apart);
/// the attention over a key mask with holes, including a query that sees no
/// key (`key_visible` is the visibility rule). One `METRIC eg2a_{kernel}_ms`
/// per kernel.
fn run_audio_kernels(warmup: usize, iters: usize) -> Result<(), String> {
    use common::{check_against, check_lane_output, fill_affine, host_rms, poisoned_f32, time_lane, upload_f32};
    use tessl::embedgemma2_audio::{
        attention, conv_out, ew, key_visible, lconv, rms_norm, sub_conv0_ln_relu, sub_conv1_ln_relu, AttentionSpec,
        Clamp, ClipTable, Ew, Lconv, RmsNorm,
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
        let feats_h = fill_affine(table.frames * mels, 51, 1.5, 0.0);
        // Some frames masked, one of them a whole 3x3 patch's rows: a kernel
        // that ignored the mask would read the features there.
        let fmask_h: Vec<u32> = (0..table.frames)
            .map(|f| u32::from(f % 7 != 3 && f % 11 != 4))
            .collect();
        let (w0_h, g0_h) = (fill_affine(c0 * 9, 52, 0.5, 0.0), fill_affine(c0, 53, 0.3, 1.0));
        let (w1_h, g1_h) = (fill_affine(9 * c0 * c1, 54, 0.5, 0.0), fill_affine(c1, 55, 0.3, 1.0));
        let feats = upload_f32(&rt, &feats_h)?;
        let fmask = rt.alloc_buffer_from_u32(&fmask_h)?;
        let (w0, g0) = (upload_f32(&rt, &w0_h)?, upload_f32(&rt, &g0_h)?);
        let (w1, g1) = (upload_f32(&rt, &w1_h)?, upload_f32(&rt, &g1_h)?);
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
        let mid_got = mid.read_f32()[..table.mid_rows * mels1 * c0].to_vec();
        let out_got = out.read_f32()[..rows * mels2 * c1].to_vec();
        check_lane_output("eg2a_sub_conv0_ln_relu", &mid_got)?;
        check_lane_output("eg2a_sub_conv1_ln_relu", &out_got)?;
        // conv0 from the features; conv1 from conv0's own output (read back), so a
        // conv0 error cannot hide a conv1 one or be counted twice.
        let mut want_mid = vec![0.0f64; mid_got.len()];
        let mut want_out = vec![0.0f64; out_got.len()];
        for clip in &table.clips {
            for t1 in 0..clip.mid_rows {
                for f1 in 0..mels1 {
                    let mut patch = [0.0f64; 9];
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let (r, col) = ((2 * t1 + kh).checked_sub(1), (2 * f1 + kw).checked_sub(1));
                            if let (Some(r), Some(col)) = (r, col) {
                                if r < clip.frames && col < mels && fmask_h[clip.frame0 + r] != 0 {
                                    patch[kh * 3 + kw] = f64::from(feats_h[(clip.frame0 + r) * mels + col]);
                                }
                            }
                        }
                    }
                    let conv: Vec<f64> = (0..c0)
                        .map(|co| (0..9).map(|k| f64::from(w0_h[co * 9 + k]) * patch[k]).sum())
                        .collect();
                    let at = ((clip.mid0 + t1) * mels1 + f1) * c0;
                    want_mid[at..at + c0].copy_from_slice(&host_ln_relu(&conv, &g0_h, f64::from(EPS)));
                }
            }
            for t2 in 0..clip.rows {
                for f2 in 0..mels2 {
                    let mut conv = vec![0.0f64; c1];
                    for kh in 0..3 {
                        let Some(r1) = (2 * t2 + kh).checked_sub(1).filter(|&r| r < clip.mid_rows) else {
                            continue;
                        };
                        // The first layer's row r1 sits at frame 2 * r1 (`mask[::2]`).
                        if fmask_h[clip.frame0 + 2 * r1] == 0 {
                            continue;
                        }
                        for kw in 0..3 {
                            let Some(f) = (2 * f2 + kw).checked_sub(1).filter(|&f| f < mels1) else {
                                continue;
                            };
                            let src = ((clip.mid0 + r1) * mels1 + f) * c0;
                            for ci in 0..c0 {
                                let a = f64::from(mid_got[src + ci]);
                                for (co, acc) in conv.iter_mut().enumerate() {
                                    *acc += a * f64::from(w1_h[((kh * 3 + kw) * c0 + ci) * c1 + co]);
                                }
                            }
                        }
                    }
                    let at = ((clip.row0 + t2) * mels2 + f2) * c1;
                    want_out[at..at + c1].copy_from_slice(&host_ln_relu(&conv, &g1_h, f64::from(EPS)));
                }
            }
        }
        check_against("eg2a_sub_conv0_ln_relu", &mid_got, &want_mid, 1e-4, 1e-3)?;
        check_against("eg2a_sub_conv1_ln_relu", &out_got, &want_out, 1e-4, 1e-3)?;
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
        let (q_h, k_h, v_h) = (
            fill_affine(rows * hidden, 62, 2.0, 0.0),
            fill_affine(rows * hidden, 63, 2.0, 0.0),
            fill_affine(rows * hidden, 64, 2.0, 0.0),
        );
        let rel_h = fill_affine(rel_rows * hidden, 65, 0.5, 0.0);
        let q_mult_h = fill_affine(head_dim, 66, 0.3, 0.6);
        // Rows 2..=5 of clip 0 masked: query 5 then sees no key (the uniform-weight
        // branch) and queries 3..=4 see fewer than `past`; row 12 of clip 1 too.
        let omask_h: Vec<bool> = (0..rows).map(|r| !(2..=5).contains(&r) && r != 12).collect();
        let (q, k, v) = (upload_f32(&rt, &q_h)?, upload_f32(&rt, &k_h)?, upload_f32(&rt, &v_h)?);
        let rel = upload_f32(&rt, &rel_h)?;
        let q_mult = upload_f32(&rt, &q_mult_h)?;
        let omask = rt.alloc_buffer_from_u32(&omask_h.iter().map(|&m| u32::from(m)).collect::<Vec<_>>())?;
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
        // Per (query, head): the window of `ctx` slots, the visible ones by
        // `key_visible`, logit softcap(ac + bd), softmax, weighted clamped values.
        let (q_mult64, k_scale) = (as_f64(&q_mult_h), f64::from(spec.k_scale));
        let cap = f64::from(spec.cap);
        let mut want = vec![0.0f64; rows * hidden];
        for clip in &table.clips {
            let clip_mask = &omask_h[clip.row0..clip.row0 + clip.rows];
            for t in 0..clip.rows {
                let (blk, i) = (t / chunk, t % chunk);
                for h in 0..heads {
                    let col0 = h * head_dim;
                    let qs: Vec<f64> = (0..head_dim)
                        .map(|d| clamp(f64::from(q_h[(clip.row0 + t) * hidden + col0 + d]), spec.q) * q_mult64[d])
                        .collect();
                    let slot = |j: usize| (blk * chunk + j) as i64 - past as i64;
                    let mut logits = Vec::new();
                    for j in 0..ctx {
                        let kp = slot(j);
                        if !key_visible(t, kp, past, clip.rows, clip_mask) {
                            continue;
                        }
                        let kp = kp as usize;
                        let mut s: f64 = (0..head_dim)
                            .map(|d| {
                                let kv = clamp(f64::from(k_h[(clip.row0 + kp) * hidden + col0 + d]), spec.k);
                                qs[d] * kv * k_scale
                            })
                            .sum();
                        // `rel[j - i]`, the relative-position term, when the table has the row.
                        if j >= i && j - i < rel_rows {
                            s += (0..head_dim)
                                .map(|d| qs[d] * f64::from(rel_h[(j - i) * hidden + col0 + d]))
                                .sum::<f64>();
                        }
                        logits.push((kp, (s / cap).tanh() * cap));
                    }
                    let mut acc = vec![0.0f64; head_dim];
                    let value =
                        |kp: usize, d: usize| clamp(f64::from(v_h[(clip.row0 + kp) * hidden + col0 + d]), spec.v);
                    if logits.is_empty() {
                        // Nothing visible: the uniform weight over the window slots inside the clip.
                        for kp in (0..ctx).map(slot).filter(|&kp| kp >= 0 && (kp as usize) < clip.rows) {
                            for (d, a) in acc.iter_mut().enumerate() {
                                *a += value(kp as usize, d) / ctx as f64;
                            }
                        }
                    } else {
                        let m = logits.iter().map(|&(_, s)| s).fold(f64::NEG_INFINITY, f64::max);
                        let den: f64 = logits.iter().map(|&(_, s)| (s - m).exp()).sum();
                        for &(kp, s) in &logits {
                            for (d, a) in acc.iter_mut().enumerate() {
                                *a += (s - m).exp() / den * value(kp, d);
                            }
                        }
                    }
                    for (d, a) in acc.into_iter().enumerate() {
                        want[(clip.row0 + t) * hidden + col0 + d] = clamp(a, spec.out);
                    }
                }
            }
        }
        check_against(lane, &got[..rows * hidden], &want, 1e-4, 1e-3)?;
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
