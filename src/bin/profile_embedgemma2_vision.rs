//! Where an `EmbedGemma2VisionModel::encode_images` call's GPU time goes, per
//! stage, on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//!   PROFILE_JSON=bench/results/embedgemma2_vision_profile_m5pro.json \
//!   cargo run --release --bin profile_embedgemma2_vision
//! ```
//!
//! A replica of `EmbedGemma2VisionModel`'s forward built from the library's
//! public kernels, loaded from the real checkpoint (all 16 layers, so the
//! weights cycle through the caches as they do in a call). The replica runs in
//! async mode, `reps` calls queued per host wait, and each stage is a *class*
//! that a bit mask switches on or off. Per workload it times, interleaved and
//! min-of-`PROFILE_ROUNDS`:
//!
//! * `full`: every class on. Its soft tokens are compared bit for bit with
//!   `encode_images`' on the same images (a replica that does not reproduce
//!   the library is not measuring it);
//! * `only:<class>`: one class on, the rest skipped: that stage's cost alone,
//!   with the real shapes, the real weights of all 16 layers and the same
//!   dependency barriers between dispatches;
//! * `skip:<class>`: the full forward without it. `full - skip` is the stage's
//!   marginal cost in the pipelined forward;
//! * `none`: no dispatch, one empty wait (the harness floor);
//! * `encode`: `encode_images` itself, upload and readback included.
//!
//! A skipped stage leaves stale data in its output buffers; GPU time does not
//! depend on values, and every buffer is zero-initialized so a skipped stage
//! can never feed a NaN downstream. A workload is an image count (default
//! `1,8` from `BENCH_WORKLOADS`): the goldens' two real processor outputs
//! (2304 and 2376 real patches of 2520), cycled, cut into forwards exactly as
//! `encode_images` cuts them (`MAX_BATCH_PATCHES`, longest first). Before every
//! sample the harness waits (at most `PROFILE_WAIT_S` seconds in all, default
//! 1800) until no other test or bench binary of a tessl checkout is running,
//! retakes a sample such a process appeared during, and flags what it could not
//! make clean; the statistics use the clean samples. Processes it cannot see
//! (display, browsers, other repositories) remain, and the `ioreg` GPU
//! utilization at the start and end of each workload records them.
//! `PROFILE_SKIP=0` leaves out the `skip:` variants. Nothing here changes
//! library code.

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tessl::embedgemma2::{encoder_attn, EncoderAttnDims};
use tessl::embedgemma2_vision::{
    axial_qkv_norm_rope, patch_position_embed_add, plan_pooling, pool_by_members, AxialQkvShape,
    EmbedGemma2VisionModel, ImageInput, VisionConfig, MAX_BATCH_PATCHES, MAX_IMAGES,
};
use tessl::gemm::{gemm, GemmBackend};
use tessl::nn::{rms_norm_f32, rms_norm_residual_add_f32};
use tessl::npy::read_npy;
use tessl::qwen35::{self, AttnTargets, Cols, GatedAct, OutCols};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{DType, GpuBuffer, Tensor};
use tessl::GpuRuntime;

use common::{
    clean_or_all, env_usize, gated, gpu_utilization, jarr, jbools, jnum, jstr, median_of, min_of, round_order,
};

const BACKEND: GemmBackend = GemmBackend::TensorOps;
const TOWER: &str = "vision_tower.";
const EMBED: &str = "embed_vision.";

// ---------------------------------------------------------------------------
// Stage classes
// ---------------------------------------------------------------------------

/// One switchable stage of the forward. `forward` runs a class only when its bit
/// of the mask is set.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum C {
    PatchGemm,
    PosEmbed,
    RmsNorm,
    QkvGemm,
    QkvNormRope,
    Attention,
    OGemm,
    NormResid,
    GateUpGemm,
    GeluMlp,
    DownGemm,
    Pool,
    EmbedNorm,
    EmbedGemm,
}

const NAMES: [&str; 14] = [
    "patch embed GEMM",
    "pos-embed add",
    "rms_norm (input, pre-ffn)",
    "qkv GEMM",
    "qk-norm+axial rope+kv store",
    "attention (hd64)",
    "o GEMM",
    "norm+residual (post-attn, post-ffn)",
    "gate|up GEMM",
    "gelu*up (mlp)",
    "down GEMM",
    "pool (3x3 by position)",
    "embedder norm",
    "embedder GEMM",
];
const N_CLASSES: usize = NAMES.len();
const ALL: u32 = (1 << N_CLASSES) - 1;

fn bit(c: C) -> u32 {
    1 << (c as usize)
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

struct LayerW {
    input_norm: GpuBuffer,
    post_attn_norm: GpuBuffer,
    pre_ffn_norm: GpuBuffer,
    post_ffn_norm: GpuBuffer,
    w_qkv: Tensor,
    w_o: Tensor,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
    gate_up: Tensor,
    down: Tensor,
}

struct Weights {
    patch_proj: Tensor,
    pos_table: Tensor,
    layers: Vec<LayerW>,
    embed_proj: Tensor,
    ones: GpuBuffer,
}

fn norm_buf(rt: &Arc<GpuRuntime>, st: &SafeTensors, name: &str, dim: usize) -> Result<GpuBuffer, String> {
    let (shape, data) = st.read_f32(&format!("{TOWER}{name}"))?;
    if shape != [dim] {
        return Err(format!("{name}: shape {shape:?}, expected [{dim}]"));
    }
    let b = rt.alloc_buffer_hot(dim.max(1) * 4)?;
    b.try_write_f32(&data)?;
    Ok(b)
}

/// `nn.Linear` weights `[out_i, in]` packed side by side into `[in, sum(out_i)]`,
/// widened to f32.
fn pack_linear(
    rt: &Arc<GpuRuntime>,
    st: &SafeTensors,
    prefix: &str,
    parts: &[(String, usize)],
    inf: usize,
) -> Result<Tensor, String> {
    let total: usize = parts.iter().map(|p| p.1).sum();
    let t = rt.alloc_tensor_f32_hot(&[inf, total])?;
    let mut dst = t.buffer.try_contents_f32()?;
    let mut col0 = 0;
    for (name, out) in parts {
        let (shape, w) = st.read_f32(&format!("{prefix}{name}"))?;
        if shape != [*out, inf] {
            return Err(format!("{name}: shape {shape:?}, expected [{out}, {inf}]"));
        }
        for o in 0..*out {
            for i in 0..inf {
                dst[i * total + col0 + o] = w[o * inf + i];
            }
        }
        col0 += out;
    }
    drop(dst);
    Ok(t)
}

impl Weights {
    fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, cfg: &VisionConfig) -> Result<Self, String> {
        let (h, inter, pd, text_h) = (
            cfg.hidden as usize,
            cfg.intermediate as usize,
            cfg.patch_dim() as usize,
            cfg.text_hidden as usize,
        );
        let qw = (cfg.heads * cfg.head_dim) as usize;
        let d = cfg.head_dim as usize;
        let one = |n: &str, out: usize| vec![(n.to_string(), out)];
        let patch_proj = pack_linear(rt, st, TOWER, &one("patch_embedder.input_proj.weight", h), pd)?;
        let size = cfg.position_embedding_size as usize;
        let pos_table = rt.alloc_tensor_bf16_hot(&[2, size, h])?;
        let (shape, bits) = st.read_bf16_bits(&format!("{TOWER}patch_embedder.position_embedding_table"))?;
        if shape != [2, size, h] {
            return Err(format!("position table: shape {shape:?}"));
        }
        pos_table.buffer.try_contents_u16()?.copy_from_slice(&bits);
        let mut layers = Vec::new();
        for i in 0..cfg.layers {
            let p = |s: &str| format!("encoder.layers.{i}.{s}");
            let lin = |names: &[(&str, usize)], inf: usize| {
                let parts: Vec<(String, usize)> = names.iter().map(|&(n, o)| (p(n), o)).collect();
                pack_linear(rt, st, TOWER, &parts, inf)
            };
            layers.push(LayerW {
                input_norm: norm_buf(rt, st, &p("input_layernorm.weight"), h)?,
                post_attn_norm: norm_buf(rt, st, &p("post_attention_layernorm.weight"), h)?,
                pre_ffn_norm: norm_buf(rt, st, &p("pre_feedforward_layernorm.weight"), h)?,
                post_ffn_norm: norm_buf(rt, st, &p("post_feedforward_layernorm.weight"), h)?,
                w_qkv: lin(
                    &[
                        ("self_attn.q_proj.linear.weight", qw),
                        ("self_attn.k_proj.linear.weight", qw),
                        ("self_attn.v_proj.linear.weight", qw),
                    ],
                    h,
                )?,
                w_o: lin(&[("self_attn.o_proj.linear.weight", h)], qw)?,
                q_norm: norm_buf(rt, st, &p("self_attn.q_norm.weight"), d)?,
                k_norm: norm_buf(rt, st, &p("self_attn.k_norm.weight"), d)?,
                gate_up: lin(
                    &[
                        ("mlp.gate_proj.linear.weight", inter),
                        ("mlp.up_proj.linear.weight", inter),
                    ],
                    h,
                )?,
                down: lin(&[("mlp.down_proj.linear.weight", h)], inter)?,
            });
        }
        let embed_proj = pack_linear(rt, st, EMBED, &one("embedding_projection.weight", text_h), h)?;
        let ones = rt.alloc_buffer_hot(h * 4)?;
        ones.try_write_f32(&vec![1.0f32; h])?;
        Ok(Self {
            patch_proj,
            pos_table,
            layers,
            embed_proj,
            ones,
        })
    }
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

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

/// Real patches of an image: the prefix before the `(-1, -1)` padding.
fn real_patches(positions: &[i32]) -> usize {
    let total = positions.len() / 2;
    (0..total)
        .position(|t| positions[2 * t] == -1 && positions[2 * t + 1] == -1)
        .unwrap_or(total)
}

/// `encode_images`' cut of a call into forwards: images longest first, runs of
/// at most `MAX_BATCH_PATCHES` padded patches and `MAX_IMAGES` images.
fn pack(valid: &[usize]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..valid.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(valid[i]));
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for i in order {
        match runs.last_mut() {
            Some(run) if run.len() < MAX_IMAGES && (run.len() + 1) * valid[run[0]] <= MAX_BATCH_PATCHES => run.push(i),
            _ => runs.push(vec![i]),
        }
    }
    runs
}

/// One forward's host-written inputs and its output.
struct Io {
    nb: u32,
    seq: u32,
    pix: Tensor,
    positions: GpuBuffer,
    lens: GpuBuffer,
    offsets: GpuBuffer,
    members: GpuBuffer,
    n_members: u32,
    soft: Tensor,
    /// The images of the chunk, in row order.
    chunk: Vec<usize>,
    rows_out: Vec<usize>,
}

fn make_io(
    rt: &Arc<GpuRuntime>,
    cfg: &VisionConfig,
    imgs: &[ImageInput<'_>],
    valid: &[usize],
    chunk: &[usize],
) -> Result<Io, String> {
    let pd = cfg.patch_dim() as usize;
    let seq = valid[chunk[0]];
    let rows = chunk.len() * seq;
    let mut pix = vec![0u32; rows * pd];
    let mut pos = vec![u32::MAX; rows * 2];
    let mut lens = Vec::new();
    let mut offsets = vec![0u32];
    let mut members: Vec<u32> = Vec::new();
    let mut rows_out = Vec::new();
    for (b, &i) in chunk.iter().enumerate() {
        let (img, n) = (&imgs[i], valid[i]);
        for (dst, &v) in pix[b * seq * pd..(b * seq + n) * pd]
            .iter_mut()
            .zip(&img.pixel_values[..n * pd])
        {
            *dst = (2.0f32 * (v - 0.5f32)).to_bits();
        }
        for (dst, &v) in pos[2 * b * seq..2 * (b * seq + n)]
            .iter_mut()
            .zip(&img.position_ids[..2 * n])
        {
            *dst = v as u32;
        }
        lens.push(n as u32);
        let plan = plan_pooling(img.position_ids, n, cfg.pooling_kernel)?;
        let base = members.len() as u32;
        let row0 = (b * seq) as u32;
        members.extend(plan.members.iter().map(|&m| row0 + m));
        offsets.extend(plan.offsets[1..].iter().map(|&o| base + o));
        rows_out.push(plan.cells.len());
    }
    let pooled_rows = offsets.len() - 1;
    Ok(Io {
        nb: chunk.len() as u32,
        seq: seq as u32,
        pix: Tensor::from_buffer(rt, rt.alloc_buffer_from_u32(&pix)?, &[rows, pd], DType::F32, 0)?,
        positions: rt.alloc_buffer_from_u32(&pos)?,
        lens: rt.alloc_buffer_from_u32(&lens)?,
        offsets: rt.alloc_buffer_from_u32(&offsets)?,
        n_members: members.len() as u32,
        members: rt.alloc_buffer_from_u32(&members)?,
        soft: rt.alloc_tensor_f32(&[pooled_rows, cfg.text_hidden as usize])?,
        chunk: chunk.to_vec(),
        rows_out,
    })
}

// ---------------------------------------------------------------------------
// Activations and the forward
// ---------------------------------------------------------------------------

struct Acts {
    resid: Tensor,
    x: Tensor,
    qkv: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn: Tensor,
    y: Tensor,
    gate_up: Tensor,
    pooled: Tensor,
    normed: Tensor,
}

impl Acts {
    /// Zeroed, so a skipped stage never feeds a NaN downstream.
    fn new(rt: &Arc<GpuRuntime>, cfg: &VisionConfig, rows: usize, pooled_rows: usize) -> Result<Self, String> {
        let (h, inter) = (cfg.hidden as usize, cfg.intermediate as usize);
        let qw = (cfg.heads * cfg.head_dim) as usize;
        let f = |shape: &[usize]| rt.alloc_tensor_f32(shape);
        Ok(Self {
            resid: f(&[rows, h])?,
            x: f(&[rows, h])?,
            qkv: f(&[rows, 3 * qw])?,
            q: f(&[rows, qw])?,
            k: f(&[rows, qw])?,
            v: f(&[rows, qw])?,
            attn: f(&[rows, qw.max(inter)])?,
            y: f(&[rows, h])?,
            gate_up: f(&[rows, 2 * inter])?,
            pooled: f(&[pooled_rows, h])?,
            normed: f(&[pooled_rows, h])?,
        })
    }
}

fn view(t: &Tensor, rows: u32, cols: u32) -> Result<Tensor, String> {
    t.try_view(&[rows as usize, cols as usize], 0)
}

struct Cx<'a> {
    rt: &'a Arc<GpuRuntime>,
    cfg: &'a VisionConfig,
    w: &'a Weights,
    a: &'a Acts,
    ios: &'a [Io],
}

/// Queue one forward over `io` (src/embedgemma2_vision.rs `forward`, op for
/// op) for the classes in `mask`; does not wait.
fn forward(cx: &Cx<'_>, io: &Io, mask: u32) -> Result<(), String> {
    let (rt, cfg, w, a) = (cx.rt, cx.cfg, cx.w, cx.a);
    let (nb, seq) = (io.nb, io.seq);
    let rows = nb as usize * seq as usize;
    let pooled_rows = io.soft.shape()[0] as u32;
    let (h, rows_u) = (cfg.hidden, rows as u32);
    let eps = cfg.rms_norm_eps;
    let qw = cfg.heads * cfg.head_dim;
    let inter = cfg.intermediate;
    let on = |c: C| mask & bit(c) != 0;
    let at = |t: &Tensor, cols: u32| view(t, rows_u, cols);
    let (resid, x, y) = (at(&a.resid, h)?, at(&a.x, h)?, at(&a.y, h)?);
    let (qkv, gate_up, mid) = (at(&a.qkv, 3 * qw)?, at(&a.gate_up, 2 * inter)?, at(&a.attn, inter)?);

    if on(C::PatchGemm) {
        gemm(&io.pix, &w.patch_proj, &resid, BACKEND)?;
    }
    if on(C::PosEmbed) {
        patch_position_embed_add(
            rt,
            &resid.buffer,
            &w.pos_table.buffer,
            &io.positions,
            rows_u,
            h,
            cfg.position_embedding_size,
        )?;
    }
    for layer in &w.layers {
        if on(C::RmsNorm) {
            rms_norm_f32(rt, &resid.buffer, &layer.input_norm, &x.buffer, rows_u, h, eps)?;
        }
        if on(C::QkvGemm) {
            gemm(&x, &layer.w_qkv, &qkv, BACKEND)?;
        }
        if on(C::QkvNormRope) {
            axial_qkv_norm_rope(
                rt,
                &AxialQkvShape {
                    batch: nb,
                    seq,
                    q_heads: cfg.heads,
                    kv_heads: cfg.heads,
                    head_dim: cfg.head_dim,
                },
                Cols::dense(&qkv.buffer, 3 * qw),
                qw,
                2 * qw,
                &layer.q_norm,
                &layer.k_norm,
                &AttnTargets {
                    q_out: &a.q.buffer,
                    k_cache: &a.k.buffer,
                    v_cache: &a.v.buffer,
                },
                &io.positions,
                cfg.rope_theta,
                eps,
            )?;
        }
        if on(C::Attention) {
            encoder_attn(
                rt,
                &a.q.buffer,
                &a.k.buffer,
                &a.v.buffer,
                &a.attn.buffer,
                &io.lens,
                EncoderAttnDims {
                    batch: nb,
                    seq,
                    heads: cfg.heads,
                    heads_kv: cfg.heads,
                    head_dim: cfg.head_dim,
                    window: 0,
                    scale: 1.0,
                },
            )?;
        }
        if on(C::OGemm) {
            gemm(&at(&a.attn, qw)?, &layer.w_o, &y, BACKEND)?;
        }
        if on(C::NormResid) {
            rms_norm_residual_add_f32(rt, &y.buffer, &layer.post_attn_norm, &resid.buffer, rows_u, h, eps, 1.0)?;
        }
        if on(C::RmsNorm) {
            rms_norm_f32(rt, &resid.buffer, &layer.pre_ffn_norm, &x.buffer, rows_u, h, eps)?;
        }
        if on(C::GateUpGemm) {
            gemm(&x, &layer.gate_up, &gate_up, BACKEND)?;
        }
        if on(C::GeluMlp) {
            let window = |off: u32| Cols {
                buf: &gate_up.buffer,
                ld: 2 * inter,
                off,
            };
            qwen35::gated_act(
                rt,
                GatedAct::GeluTanh,
                window(0),
                window(inter),
                OutCols {
                    cols: Cols::dense(&mid.buffer, inter),
                    dtype: DType::F32,
                },
                rows_u,
                inter,
            )?;
        }
        if on(C::DownGemm) {
            gemm(&mid, &layer.down, &y, BACKEND)?;
        }
        if on(C::NormResid) {
            rms_norm_residual_add_f32(rt, &y.buffer, &layer.post_ffn_norm, &resid.buffer, rows_u, h, eps, 1.0)?;
        }
    }
    let kk = (cfg.pooling_kernel * cfg.pooling_kernel) as f32;
    let (pooled, normed) = (view(&a.pooled, pooled_rows, h)?, view(&a.normed, pooled_rows, h)?);
    let soft = view(&io.soft, pooled_rows, cfg.text_hidden)?;
    if on(C::Pool) {
        pool_by_members(
            rt,
            &resid.buffer,
            &io.offsets,
            &io.members,
            &pooled.buffer,
            pooled_rows,
            rows_u,
            h,
            io.n_members,
            1.0f32 / kk,
            (h as f32).sqrt(),
        )?;
    }
    if on(C::EmbedNorm) {
        rms_norm_f32(rt, &pooled.buffer, &w.ones, &normed.buffer, pooled_rows, h, eps)?;
    }
    if on(C::EmbedGemm) {
        gemm(&normed, &w.embed_proj, &soft, BACKEND)?;
    }
    Ok(())
}

/// Every forward of the call, in order.
fn call(cx: &Cx<'_>, mask: u32) -> Result<(), String> {
    for io in cx.ios {
        forward(cx, io, mask)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Costs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Cost {
    flops: f64,
    /// A lower bound on bytes moved: each operand read or written once.
    bytes: f64,
}

fn gemm_cost(m: usize, k: usize, n: usize, count: usize) -> Cost {
    Cost {
        flops: 2.0 * m as f64 * k as f64 * n as f64 * count as f64,
        bytes: 4.0 * (k * n + m * k + m * n) as f64 * count as f64,
    }
}

fn class_costs(cfg: &VisionConfig, ios: &[Io]) -> [Cost; N_CLASSES] {
    let (h, inter, pd) = (cfg.hidden as usize, cfg.intermediate as usize, cfg.patch_dim() as usize);
    let (l, qw, d, heads) = (
        cfg.layers as usize,
        (cfg.heads * cfg.head_dim) as usize,
        cfg.head_dim as usize,
        cfg.heads as usize,
    );
    let mut c = [Cost::default(); N_CLASSES];
    let el = |n: usize| 4.0 * n as f64;
    let add = |dst: &mut Cost, x: Cost| {
        dst.flops += x.flops;
        dst.bytes += x.bytes;
    };
    for io in ios {
        let (nb, seq) = (io.nb as usize, io.seq as usize);
        let rows = nb * seq;
        let s = io.soft.shape()[0];
        add(&mut c[C::PatchGemm as usize], gemm_cost(rows, pd, h, 1));
        add(
            &mut c[C::PosEmbed as usize],
            Cost {
                flops: 0.0,
                bytes: el(2 * rows * h) + 2.0 * 2.0 * (rows * h) as f64,
            },
        );
        add(
            &mut c[C::RmsNorm as usize],
            Cost {
                flops: 0.0,
                bytes: 2.0 * l as f64 * el(2 * rows * h),
            },
        );
        add(&mut c[C::QkvGemm as usize], gemm_cost(rows, h, 3 * qw, l));
        add(
            &mut c[C::QkvNormRope as usize],
            Cost {
                flops: 0.0,
                bytes: l as f64 * 2.0 * el(rows * 3 * qw),
            },
        );
        add(
            &mut c[C::Attention as usize],
            Cost {
                flops: l as f64 * 4.0 * (nb * heads * d) as f64 * (seq * seq) as f64,
                bytes: l as f64 * el(4 * rows * qw),
            },
        );
        add(&mut c[C::OGemm as usize], gemm_cost(rows, qw, h, l));
        add(
            &mut c[C::NormResid as usize],
            Cost {
                flops: 0.0,
                bytes: 2.0 * l as f64 * el(3 * rows * h),
            },
        );
        add(&mut c[C::GateUpGemm as usize], gemm_cost(rows, h, 2 * inter, l));
        add(
            &mut c[C::GeluMlp as usize],
            Cost {
                flops: 0.0,
                bytes: l as f64 * el(rows * 3 * inter),
            },
        );
        add(&mut c[C::DownGemm as usize], gemm_cost(rows, inter, h, l));
        add(
            &mut c[C::Pool as usize],
            Cost {
                flops: 0.0,
                bytes: el(rows * h) + el(s * h),
            },
        );
        add(
            &mut c[C::EmbedNorm as usize],
            Cost {
                flops: 0.0,
                bytes: el(2 * s * h),
            },
        );
        add(
            &mut c[C::EmbedGemm as usize],
            gemm_cost(s, h, cfg.text_hidden as usize, 1),
        );
    }
    c
}

// ---------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------

struct Variant {
    name: String,
    class: Option<usize>,
    kind: &'static str,
    mask: u32,
    samples_ms: Vec<f64>,
    enc_ms: Vec<f64>,
    raw_ms: Vec<f64>,
    raw_enc_ms: Vec<f64>,
    contended: Vec<bool>,
    resampled: usize,
    dispatches: usize,
}

/// `reps` calls queued, one wait: (total ms per call, host encode ms per call).
fn timed_reps(rt: &Arc<GpuRuntime>, cx: &Cx<'_>, mask: u32, reps: usize) -> Result<(f64, f64), String> {
    let t0 = Instant::now();
    for _ in 0..reps {
        call(cx, mask)?;
    }
    let enc = t0.elapsed();
    rt.synchronize()?;
    let tot = t0.elapsed();
    Ok((
        tot.as_secs_f64() * 1e3 / reps as f64,
        enc.as_secs_f64() * 1e3 / reps as f64,
    ))
}

fn main() -> Result<(), String> {
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let refs = PathBuf::from(
        std::env::var("EMBEDGEMMA2_MM_REF_DIR")
            .map_err(|_| "EMBEDGEMMA2_MM_REF_DIR must name the make_mm_reference.py output directory")?,
    );
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| "1,8".into());
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
    let rounds = env_usize("PROFILE_ROUNDS", 5, 1)?;
    let with_skip = std::env::var("PROFILE_SKIP").as_deref() != Ok("0");
    let out_path = std::env::var("PROFILE_JSON").unwrap_or_else(|_| "target/profile_vision.json".into());

    let cfg = VisionConfig::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let goldens = [golden(&refs, 0)?, golden(&refs, 1)?];
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(true)?;
    let model = EmbedGemma2VisionModel::load(&rt, &st, cfg.clone())?;
    let weights = Weights::load(&rt, &st, &cfg)?;
    let util0 = gpu_utilization();
    eprintln!("gpu utilization at start: {util0}%");

    let mut json_wl = Vec::new();
    for &n in &counts {
        let name = format!("{n}img");
        let imgs: Vec<ImageInput<'_>> = (0..n)
            .map(|i| {
                let g = &goldens[i % goldens.len()];
                ImageInput {
                    pixel_values: &g.pixels,
                    position_ids: &g.positions,
                }
            })
            .collect();
        let valid: Vec<usize> = imgs.iter().map(|im| real_patches(im.position_ids)).collect();
        let chunks = pack(&valid);
        let rows = chunks.iter().map(|c| c.len() * valid[c[0]]).max().unwrap_or(1);
        let pooled_rows = chunks
            .iter()
            .map(|c| {
                c.iter()
                    .map(|&i| plan_pooling(imgs[i].position_ids, valid[i], cfg.pooling_kernel).map(|p| p.cells.len()))
                    .sum::<Result<usize, String>>()
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(1);
        let acts = Acts::new(&rt, &cfg, rows, pooled_rows)?;
        let ios: Vec<Io> = chunks
            .iter()
            .map(|c| make_io(&rt, &cfg, &imgs, &valid, c))
            .collect::<Result<_, _>>()?;
        let cx = Cx {
            rt: &rt,
            cfg: &cfg,
            w: &weights,
            a: &acts,
            ios: &ios,
        };
        let costs = class_costs(&cfg, &ios);
        let total_rows: usize = ios.iter().map(|io| (io.nb * io.seq) as usize).sum();

        // Validation: the replica reproduces the library's soft tokens.
        let lib = model.encode_images(&imgs, false)?;
        timed_reps(&rt, &cx, ALL, 1)?;
        let text_h = cfg.text_hidden as usize;
        let (mut values, mut bit_equal) = (0usize, 0usize);
        for io in &ios {
            let all = io.soft.buffer.try_contents_f32()?;
            let mut row = 0usize;
            for (b, &i) in io.chunk.iter().enumerate() {
                let s = io.rows_out[b];
                let mine = &all[row * text_h..(row + s) * text_h];
                values += mine.len();
                bit_equal += mine
                    .iter()
                    .zip(&lib.soft[i])
                    .filter(|(x, y)| x.to_bits() == y.to_bits())
                    .count();
                row += s;
            }
        }
        eprintln!("{name}: replica vs encode_images: {bit_equal}/{values} soft values bit-equal");
        if bit_equal != values {
            return Err(format!("{name}: the replica forward disagrees with encode_images"));
        }

        let mut variants: Vec<Variant> = Vec::new();
        let mut mk = |name: String, class: Option<usize>, kind: &'static str, mask: u32| -> Result<(), String> {
            rt.synchronize()?;
            rt.take_dispatch_count();
            call(&cx, mask)?;
            rt.synchronize()?;
            let dispatches = rt.take_dispatch_count();
            variants.push(Variant {
                name,
                class,
                kind,
                mask,
                samples_ms: Vec::new(),
                enc_ms: Vec::new(),
                raw_ms: Vec::new(),
                raw_enc_ms: Vec::new(),
                contended: Vec::new(),
                resampled: 0,
                dispatches,
            });
            Ok(())
        };
        mk("full".into(), None, "full", ALL)?;
        mk("none".into(), None, "none", 0)?;
        for (c, nm) in NAMES.iter().enumerate() {
            mk(format!("only:{nm}"), Some(c), "only", 1 << c)?;
            if with_skip {
                mk(format!("skip:{nm}"), Some(c), "skip", ALL & !(1 << c))?;
            }
        }
        let dispatch_full = variants[0].dispatches;

        let probe = {
            timed_reps(&rt, &cx, ALL, 1)?;
            timed_reps(&rt, &cx, ALL, 1)?.0
        };
        let reps = ((200.0 / probe).ceil() as usize)
            .clamp(1, 8)
            .min((8000 / dispatch_full.max(1)).max(1));
        eprintln!(
            "{name}: {} forward(s), {dispatch_full} dispatches per call, {reps} call(s) per wait, probe {probe:.2} ms",
            ios.len()
        );

        let n_var = variants.len() + 1; // + encode_images
        for v in &variants {
            timed_reps(&rt, &cx, v.mask, reps.min(2))?;
        }
        model.encode_images(&imgs, false)?;
        let mut encode_raw = Vec::with_capacity(rounds);
        let mut encode_contended = Vec::with_capacity(rounds);
        for round in 0..rounds {
            for i in round_order(round, n_var) {
                if i == variants.len() {
                    let (ms, contended, _) = gated(|| {
                        let t0 = Instant::now();
                        model.encode_images(&imgs, false)?;
                        Ok(t0.elapsed().as_secs_f64() * 1e3)
                    })?;
                    encode_raw.push(ms);
                    encode_contended.push(contended);
                } else {
                    let mask = variants[i].mask;
                    let ((tot, enc), contended, retakes) = gated(|| timed_reps(&rt, &cx, mask, reps))?;
                    let v = &mut variants[i];
                    v.raw_ms.push(tot);
                    v.raw_enc_ms.push(enc);
                    v.contended.push(contended);
                    v.resampled += retakes;
                }
            }
        }
        for v in &mut variants {
            v.samples_ms = clean_or_all(&v.raw_ms, &v.contended);
            v.enc_ms = clean_or_all(&v.raw_enc_ms, &v.contended);
        }
        let encode_samples = clean_or_all(&encode_raw, &encode_contended);
        let n_contended: usize = variants
            .iter()
            .map(|v| v.contended.iter().filter(|&&c| c).count())
            .sum();
        let n_samples: usize = variants.iter().map(|v| v.contended.len()).sum();
        let n_resampled: usize = variants.iter().map(|v| v.resampled).sum();
        let util1 = gpu_utilization();

        // ---- report ----
        let full = min_of(&variants[0].samples_ms);
        let none = min_of(&variants[1].samples_ms);
        let enc_min = min_of(&encode_samples);
        println!(
            "== workload {name} ({} forward(s), {total_rows} padded rows, {dispatch_full} dispatches/call, {reps} calls per wait, {rounds} rounds)",
            ios.len()
        );
        println!(
            "   full replica {full:.3} ms (min) / {:.3} (median); encode_images() {enc_min:.3} ms (min); empty wait {none:.3} ms; host encode time of full {:.3} ms",
            median_of(&variants[0].samples_ms),
            min_of(&variants[0].enc_ms)
        );
        println!(
            "   {:<38} {:>5} {:>9} {:>6} {:>10} {:>6} {:>9} {:>8} {:>8}",
            "class", "disp", "only ms", "%full", "full-skip", "%full", "us/disp", "GB/s", "TFLOP/s"
        );
        let mut order: Vec<usize> = (0..N_CLASSES).collect();
        let only_ms = |c: usize| {
            min_of(
                &variants
                    .iter()
                    .find(|v| v.kind == "only" && v.class == Some(c))
                    .unwrap()
                    .samples_ms,
            )
        };
        let skip_ms = |c: usize| {
            variants
                .iter()
                .find(|v| v.kind == "skip" && v.class == Some(c))
                .map(|v| min_of(&v.samples_ms))
        };
        order.sort_by(|&x, &y| only_ms(y).total_cmp(&only_ms(x)));
        let (mut sum_only, mut sum_marg) = (0.0, 0.0);
        for &c in &order {
            let o = only_ms(c);
            let v = variants
                .iter()
                .find(|v| v.kind == "only" && v.class == Some(c))
                .unwrap();
            sum_only += o;
            let marg = skip_ms(c).map(|s| full - s);
            sum_marg += marg.unwrap_or(0.0);
            let work = costs[c];
            let gbs = if o > 0.0 { work.bytes / (o * 1e-3) / 1e9 } else { 0.0 };
            let tf = if work.flops > 0.0 {
                work.flops / (o * 1e-3) / 1e12
            } else {
                0.0
            };
            println!(
                "   {:<38} {:>5} {:>9.3} {:>5.1}% {:>10} {:>6} {:>9.2} {:>8.0} {:>8}",
                NAMES[c],
                v.dispatches,
                o,
                100.0 * o / full,
                marg.map_or("-".to_string(), |m| format!("{m:.3}")),
                marg.map_or("-".to_string(), |m| format!("{:.1}%", 100.0 * m / full)),
                1e3 * o / v.dispatches.max(1) as f64,
                gbs,
                if tf > 0.0 { format!("{tf:.2}") } else { "-".into() }
            );
        }
        let marg_note = if variants.iter().any(|v| v.kind == "skip") {
            format!(
                "sum(full-skip) {sum_marg:.3} ms = {:.1}% of full",
                100.0 * sum_marg / full
            )
        } else {
            "no skip variants (PROFILE_SKIP=0)".to_string()
        };
        println!(
            "   sum(only) {sum_only:.3} ms = {:.1}% of full; {marg_note}; gpu util {util0}% -> {util1}%",
            100.0 * sum_only / full
        );
        println!(
            "   samples: {n_samples} kept, {n_contended} with another tessl GPU process seen (statistics use the clean ones), {n_resampled} retaken"
        );

        // ---- json ----
        let var_json: Vec<String> = variants
            .iter()
            .map(|v| {
                format!(
                    "{{\"name\":{},\"kind\":{},\"class\":{},\"mask\":{},\"dispatches\":{},\"ms_per_call\":{},\"host_encode_ms_per_call\":{},\"ms_min\":{},\"raw_ms_per_call\":{},\"contended\":{},\"resampled\":{}}}",
                    jstr(&v.name),
                    jstr(v.kind),
                    v.class.map_or("null".to_string(), |c| c.to_string()),
                    v.mask,
                    v.dispatches,
                    jarr(&v.samples_ms),
                    jarr(&v.enc_ms),
                    jnum(min_of(&v.samples_ms)),
                    jarr(&v.raw_ms),
                    jbools(&v.contended),
                    v.resampled
                )
            })
            .collect();
        let cost_json: Vec<String> = (0..N_CLASSES)
            .map(|c| {
                format!(
                    "{{\"class\":{},\"flops\":{:.0},\"bytes_min\":{:.0}}}",
                    jstr(NAMES[c]),
                    costs[c].flops,
                    costs[c].bytes
                )
            })
            .collect();
        json_wl.push(format!(
            "{{\"workload\":{},\"images\":{n},\"forwards\":{},\"padded_rows\":{total_rows},\"reps_per_wait\":{reps},\"rounds\":{rounds},\"dispatches_full\":{dispatch_full},\"replica_vs_encode_images\":{{\"values\":{values},\"bit_equal\":{bit_equal}}},\"encode_images_ms\":{},\"encode_images_raw_ms\":{},\"encode_images_contended\":{},\"samples_contended\":{n_contended},\"samples_total\":{n_samples},\"samples_resampled\":{n_resampled},\"gpu_util_start_pct\":{},\"gpu_util_end_pct\":{},\"costs\":[{}],\"variants\":[{}]}}",
            jstr(&name),
            ios.len(),
            jarr(&encode_samples),
            jarr(&encode_raw),
            jbools(&encode_contended),
            jstr(&util0),
            jstr(&util1),
            cost_json.join(","),
            var_json.join(",")
        ));
    }
    let body = format!(
        "{{\"mode\":\"stages\",\"classes\":[{}],\"workloads\":[{}]}}",
        NAMES.iter().map(|n| jstr(n)).collect::<Vec<_>>().join(","),
        json_wl.join(",")
    );
    common::atomic_write(std::path::Path::new(&out_path), body.as_bytes())?;
    eprintln!("wrote {out_path}");
    Ok(())
}
