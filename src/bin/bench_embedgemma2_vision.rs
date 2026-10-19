//! End-to-end latency of `tessl::embedgemma2_vision` on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//!   cargo run --release --bin bench_embedgemma2_vision
//! ```
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,`; a workload is a count
//! of images (default `1,8`), one `encode_images` call. The images are the
//! goldens' two real processor outputs (`image0_*`: 224 x 224, 2304 real
//! patches; `image1_*`: 640 x 360, 2376), both padded to 2520 patches,
//! cycled to the count, so `8` is a ragged batch of four of each that
//! `encode_images` cuts into two forwards (`forwards` in the output).
//!
//! `BENCH_ITERS` (default 10) and `BENCH_WARMUP` (default 3) are per workload.
//! `BENCH_PRECISION=f32|bf16` (default `f32`) picks the operand precision the
//! tower is loaded with ([`EmbedGemma2VisionModel::load_with`]: `GemmOperands::ExactF32`
//! or `Bf16`); the object's `backend` says which (`tessl-f32`, `tessl-bf16`).
//! Prints one JSON array on stdout, one object per workload: `{workload,
//! backend, ms_min, ms_median, iters, forwards, weight_mib, loaded_mib, device_peak_mib,
//! peak_footprint_mib}`, the format `bench_paired --format workloads` reads.
//! `weight_mib` is the model's weight bytes ([`EmbedGemma2VisionModel::weight_bytes`]);
//! `loaded_mib` is the device bytes allocated once it is loaded (those, and the
//! pool's rounding; no activations). The base build (433e659), which has neither
//! `load_with` nor `weight_bytes`, runs this file as of f4a9ece, which differs
//! from this one in those two lines and the `BENCH_PRECISION` knob only.
//! stderr carries `HASH <workload> <fnv1a64>` per workload, over the bits of
//! every soft value in call order: two builds with equal hashes produced
//! bit-identical outputs.
//!
//! `--kernels` instead runs four of the tower's kernels through their public
//! functions on random operands, with no checkpoint (`run_vision_kernels`).
//!
//! What is timed: one `encode_images` call from the host's pixel values to the
//! host's soft tokens: the pixel scaling and upload, every forward, the wait
//! and the read back. Not timed: the model load and the goldens' read.
//!
//! Before anything is timed every workload's soft tokens must be finite; the
//! run aborts otherwise, so a forward that wrote nothing cannot post the best
//! time.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, VisionConfig};
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

use common::{env_usize, peak_footprint};

const DEFAULT_WORKLOADS: &str = "1,8";
const MIB: f64 = (1u64 << 20) as f64;

/// One image's processor tensors from the goldens.
struct Golden {
    pixels: Vec<f32>,
    positions: Vec<i32>,
}

fn golden(dir: &std::path::Path, n: usize) -> Result<Golden, String> {
    let read = |name: String| read_npy(&dir.join(&name)).map_err(|e| format!("{name}: {e}"));
    let pixels = read(format!("image{n}_pixel_values.npy"))?;
    let positions = read(format!("image{n}_image_position_ids.npy"))?;
    let positions = positions
        .data_i64
        .as_ref()
        .ok_or("image position ids are not i64")?
        .iter()
        .map(|&v| i32::try_from(v).map_err(|_| format!("position {v} exceeds i32")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Golden {
        pixels: pixels.f32_slice()?.to_vec(),
        positions,
    })
}

/// `--kernels`: the vision tower's four entry points that no end-to-end lane
/// reaches without a checkpoint (`vision_pos_embed_add_f32`,
/// `vision_pool_members_f32`, `vision_qkv_norm_rope_f32`,
/// `vision_qkv_norm_rope_bf16`) through their public functions on random
/// operands, each checked once before it is timed: the first two against an
/// f64 host reference; the rope pass on what a rotation cannot change (each q
/// and k head keeps the norm of its normed input, every v head is the
/// weightless norm exactly). One `METRIC vision_{pass}_ms` per pass.
fn run_vision_kernels(warmup: usize, iters: usize) -> Result<(), String> {
    use common::{
        check_against, check_lane_output, host_rms, poisoned_bf16, poisoned_f32, read_bf16, time_lane, upload_bf16,
        upload_f32,
    };
    use tessl::embedgemma2_vision::{
        axial_qkv_norm_rope, axial_qkv_norm_rope_bf16, patch_position_embed_add, pool_by_members, AxialQkvShape,
    };
    use tessl::qwen35::{AttnTargets, Cols};
    use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16};

    let rt = GpuRuntime::new_inference()?;
    let as_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    let words = |v: &[i32]| -> Vec<u32> { v.iter().map(|&x| x as u32).collect() };

    // The patch embedder's position add: `h += table[0][x] + table[1][y]`, a
    // padding row (-1, -1) zeroed.
    {
        const ROWS: usize = 6;
        const DIM: usize = 24;
        const TABLE_LEN: usize = 7;
        let lane = "vision_pos_embed_add_f32";
        let h0 = common::fill_affine(ROWS * DIM, 41, 2.0, 0.0);
        let table_f = common::fill_affine(2 * TABLE_LEN * DIM, 42, 1.0, 0.0);
        let table_w: Vec<f32> = f32_slice_to_bf16(&table_f).into_iter().map(bf16_bits_to_f32).collect();
        let pos: Vec<i32> = vec![0, 0, 1, 2, 6, 3, 2, 6, 5, 1, -1, -1];
        let want: Vec<f64> = (0..ROWS)
            .flat_map(|r| {
                let (x, y) = (pos[2 * r], pos[2 * r + 1]);
                (0..DIM)
                    .map(|c| {
                        if x < 0 || y < 0 {
                            0.0
                        } else {
                            f64::from(h0[r * DIM + c])
                                + f64::from(table_w[x as usize * DIM + c])
                                + f64::from(table_w[(TABLE_LEN + y as usize) * DIM + c])
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let (h, table, positions) = (
            upload_f32(&rt, &h0)?,
            upload_bf16(&rt, &table_f)?,
            rt.alloc_buffer_from_u32(&words(&pos))?,
        );
        patch_position_embed_add(&rt, &h, &table, &positions, ROWS as u32, DIM as u32, TABLE_LEN as u32)?;
        rt.synchronize()?;
        let got = h.read_f32();
        check_lane_output(lane, &got[..ROWS * DIM])?;
        check_against(lane, &got[..ROWS * DIM], &want, 1e-6, 1e-6)?;
        time_lane(&rt, lane, warmup, iters, || {
            patch_position_embed_add(&rt, &h, &table, &positions, ROWS as u32, DIM as u32, TABLE_LEN as u32)
        })?;
    }

    // The pooler's cell sums, one cell empty (exactly +0).
    {
        const ROWS: usize = 9;
        const DIM: usize = 20;
        const WEIGHT: f32 = 0.25;
        const SCALE: f32 = 4.5;
        let lane = "vision_pool_members_f32";
        let h0 = common::fill_affine(ROWS * DIM, 43, 2.0, 0.0);
        let offsets: Vec<u32> = vec![0, 4, 4, 7, 9];
        let members: Vec<u32> = vec![0, 1, 3, 4, 8, 2, 6, 5, 7];
        let out_rows = offsets.len() - 1;
        let want: Vec<f64> = (0..out_rows)
            .flat_map(|s| {
                let cell = &members[offsets[s] as usize..offsets[s + 1] as usize];
                (0..DIM)
                    .map(|c| {
                        cell.iter().map(|&m| f64::from(h0[m as usize * DIM + c])).sum::<f64>()
                            * f64::from(WEIGHT)
                            * f64::from(SCALE)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let (h, off_b, mem_b, out) = (
            upload_f32(&rt, &h0)?,
            rt.alloc_buffer_from_u32(&offsets)?,
            rt.alloc_buffer_from_u32(&members)?,
            poisoned_f32(&rt, out_rows * DIM)?,
        );
        let dispatch = || {
            pool_by_members(
                &rt,
                &h,
                &off_b,
                &mem_b,
                &out,
                out_rows as u32,
                ROWS as u32,
                DIM as u32,
                members.len() as u32,
                WEIGHT,
                SCALE,
            )
        };
        dispatch()?;
        rt.synchronize()?;
        let got = out.read_f32();
        check_lane_output(lane, &got[..out_rows * DIM])?;
        check_against(lane, &got[..out_rows * DIM], &want, 1e-5, 1e-5)?;
        time_lane(&rt, lane, warmup, iters, dispatch)?;
    }

    // Q/K norm with weight, axial RoPE, weightless V norm and the K/V store.
    {
        let shape = AxialQkvShape {
            batch: 2,
            seq: 5,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 16,
        };
        let (b, t, qh, kvh, d) = (
            shape.batch as usize,
            shape.seq as usize,
            shape.q_heads as usize,
            shape.kv_heads as usize,
            shape.head_dim as usize,
        );
        const EPS: f32 = 1e-6;
        const THETA: f32 = 100.0;
        let rows = b * t;
        let width = (qh + 2 * kvh) * d;
        let (k_col, v_col) = ((qh * d) as u32, ((qh + kvh) * d) as u32);
        let proj_h = common::fill_affine(rows * width, 44, 3.0, 0.0);
        let q_w = common::fill_affine(d, 45, 0.4, 1.0);
        let k_w = common::fill_affine(d, 46, 0.4, 1.0);
        // Patch (x, y) in a 3-wide grid; the last patch of each image is padding.
        let pos: Vec<i32> = (0..rows)
            .flat_map(|r| {
                let i = (r % t) as i32;
                if i as usize == t - 1 {
                    [-1, -1]
                } else {
                    [i % 3, i / 3]
                }
            })
            .collect();
        let (proj, qn, kn, positions) = (
            upload_f32(&rt, &proj_h)?,
            upload_f32(&rt, &q_w)?,
            upload_f32(&rt, &k_w)?,
            rt.alloc_buffer_from_u32(&words(&pos))?,
        );
        let norm_of = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        // The head `head` (of `heads`) of projection row `r` at column `col0`, normed.
        let normed = |r: usize, col0: usize, head: usize, w: Option<&[f32]>| -> Vec<f64> {
            let at = r * width + col0 + head * d;
            host_rms(&as_f64(&proj_h[at..at + d]), w, f64::from(EPS))
        };
        for bf16 in [false, true] {
            let lane = if bf16 {
                "vision_qkv_norm_rope_bf16"
            } else {
                "vision_qkv_norm_rope_f32"
            };
            let make = |n: usize| {
                if bf16 {
                    poisoned_bf16(&rt, n)
                } else {
                    poisoned_f32(&rt, n)
                }
            };
            let (q_out, k_cache, v_cache) = (make(rows * qh * d)?, make(rows * kvh * d)?, make(rows * kvh * d)?);
            let dispatch = || {
                let targets = AttnTargets {
                    q_out: &q_out,
                    k_cache: &k_cache,
                    v_cache: &v_cache,
                };
                let f = if bf16 {
                    axial_qkv_norm_rope_bf16
                } else {
                    axial_qkv_norm_rope
                };
                f(
                    &rt,
                    &shape,
                    Cols::dense(&proj, width as u32),
                    k_col,
                    v_col,
                    &qn,
                    &kn,
                    &targets,
                    &positions,
                    THETA,
                    EPS,
                )
            };
            dispatch()?;
            rt.synchronize()?;
            let read = |buf: &tessl::tensor::GpuBuffer, n: usize| -> Result<Vec<f32>, String> {
                Ok(if bf16 {
                    read_bf16(&rt, buf, n)?
                } else {
                    buf.read_f32()[..n].to_vec()
                })
            };
            let (q, k, v) = (
                read(&q_out, rows * qh * d)?,
                read(&k_cache, rows * kvh * d)?,
                read(&v_cache, rows * kvh * d)?,
            );
            // The padding rows' stores are not asserted; every real row's are.
            let (tol_abs, tol_rel) = if bf16 { (1e-4, 1.5e-2) } else { (1e-5, 1e-4) };
            let mut checked = Vec::new();
            for r in (0..rows).filter(|r| r % t != t - 1) {
                for h in 0..qh {
                    let want = norm_of(&normed(r, 0, h, Some(q_w.as_slice())));
                    let got = norm_of(&as_f64(&q[(r * qh + h) * d..(r * qh + h + 1) * d]));
                    check_against(
                        &format!("{lane} q norm row {r} head {h}"),
                        &[got as f32],
                        &[want],
                        tol_abs,
                        tol_rel,
                    )?;
                    checked.extend_from_slice(&q[(r * qh + h) * d..(r * qh + h + 1) * d]);
                }
                for h in 0..kvh {
                    let want = norm_of(&normed(r, k_col as usize, h, Some(k_w.as_slice())));
                    let got = norm_of(&as_f64(&k[(r * kvh + h) * d..(r * kvh + h + 1) * d]));
                    check_against(
                        &format!("{lane} k norm row {r} head {h}"),
                        &[got as f32],
                        &[want],
                        tol_abs,
                        tol_rel,
                    )?;
                    let want_v = normed(r, v_col as usize, h, None);
                    check_against(
                        &format!("{lane} v row {r} head {h}"),
                        &v[(r * kvh + h) * d..(r * kvh + h + 1) * d],
                        &want_v,
                        tol_abs,
                        tol_rel,
                    )?;
                }
            }
            check_lane_output(lane, &checked)?;
            time_lane(&rt, lane, warmup, iters, dispatch)?;
        }
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
        let outcome = common::mark_incomplete_on_error(run_vision_kernels(warmup, iters));
        common::emit_kernel_trace();
        return outcome;
    }
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let refs = PathBuf::from(
        std::env::var("EMBEDGEMMA2_MM_REF_DIR")
            .map_err(|_| "EMBEDGEMMA2_MM_REF_DIR must name the make_mm_reference.py output directory")?,
    );
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    let counts = spec
        .split(',')
        .map(|w| {
            w.trim()
                .parse::<usize>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| format!("workload {w:?} is not a positive image count"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;

    let cfg = VisionConfig::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let goldens = [golden(&refs, 0)?, golden(&refs, 1)?];
    let rt = GpuRuntime::new_inference()?;
    let (operands, backend) = match std::env::var("BENCH_PRECISION").as_deref() {
        Err(_) | Ok("f32") => (GemmOperands::ExactF32, "tessl-f32"),
        Ok("bf16") => (GemmOperands::Bf16, "tessl-bf16"),
        Ok(other) => return Err(format!("BENCH_PRECISION={other:?}: expected f32 or bf16")),
    };
    let model = EmbedGemma2VisionModel::load_with(&rt, &st, cfg, operands)?;
    let weight_mib = model.weight_bytes() as f64 / MIB;
    let loaded = rt.current_allocated_bytes();
    let loaded_mib = loaded as f64 / MIB;

    let mut rows = Vec::new();
    for &n in &counts {
        let batch: Vec<ImageInput<'_>> = (0..n)
            .map(|i| {
                let g = &goldens[i % goldens.len()];
                ImageInput {
                    pixel_values: &g.pixels,
                    position_ids: &g.positions,
                }
            })
            .collect();
        rt.reset_peak_allocated_bytes();
        let out = model.encode_images(&batch, false)?;
        if let Some((i, _)) = out
            .soft
            .iter()
            .enumerate()
            .find(|(_, s)| s.is_empty() || !s.iter().all(|v| v.is_finite()))
        {
            return Err(format!("{n} images: image {i}'s soft tokens are empty or not finite"));
        }
        // FNV-1a over every soft value's bits, images in call order: equal
        // hashes across two builds are bit-identical outputs.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for v in out.soft.iter().flatten() {
            for byte in v.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        eprintln!("HASH {n}img {hash:016x}");
        for _ in 0..warmup {
            model.encode_images(&batch, false)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.encode_images(&batch, false)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        let device_peak = rt.peak_allocated_bytes().saturating_sub(loaded);
        rows.push(format!(
            "{{\"workload\":\"{n}img\",\"backend\":\"{backend}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"forwards\":{},\"weight_mib\":{weight_mib:.1},\"loaded_mib\":{loaded_mib:.1},\"device_peak_mib\":{:.1},\"peak_footprint_mib\":{:.1}}}",
            ms[0],
            ms[ms.len() / 2],
            out.forwards,
            device_peak as f64 / MIB,
            peak_footprint()? as f64 / MIB,
        ));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}
