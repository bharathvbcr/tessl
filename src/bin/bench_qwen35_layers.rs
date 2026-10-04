//! Prefill throughput of `tessl::qwen35` at Qwen3.5-2B's shapes.
//!
//! ```text
//! cargo run --release --bin bench_qwen35_layers                 # T = 1024, 2048, 8192
//! cargo run --release --bin bench_qwen35_layers -- 1024 4096    # chosen T
//! cargo run --release --bin bench_qwen35_layers -- --check-only # plausibility gate, no timing
//! cargo run --release --bin bench_qwen35_layers -- --attn-rows  # scalar nn::flash_attn_rows
//! cargo run --release --bin bench_qwen35_layers -- --paired-attn 200 8192
//! cargo run --release --bin bench_qwen35_layers -- --attn-tile=q64_k64_sg8
//! cargo run --release --bin bench_qwen35_layers -- --mlp-unfused # mlp_silu + cast
//! cargo run --release --bin bench_qwen35_layers -- --gdn-scan16  # 16-column GDN scan
//! ```
//!
//! Attention runs on `qwen35::attn_prefill` (the TensorOps kernel) at its
//! default tile. `--attn-tile=LABEL` picks another `qwen35::AttnTile`, and
//! `--attn-rows` selects `nn::flash_attn_rows`, the kernel it replaced. So
//! the choices can be compared in one binary on one machine state.
//! `--paired-attn` times only the prefill attention kernels, both of them,
//! in this process: same Q/K/V, ABBA order (which kernel goes first rotates
//! each round). It does not build the 24-layer model.
//!
//! Shapes are Qwen3.5-2B's `text_config` (hidden 2048, 16 key and 16 value GDN
//! heads of 128, 8 query and 2 KV attention heads of 256, rotary 64, MLP 6144,
//! vocab 248320, `full_attention_interval` 4 over 24 layers). Weights are
//! random, bf16, and distinct per layer, so no layer's weights are still in
//! cache from the one before. Batch 1, activations f32 as tessl's GEMM writes
//! them, rounded to bf16 only on the way into a GEMM.
//!
//! What is timed:
//!
//! * **forward** — all 24 layers in `layer_types` order, each with its input
//!   norm, mixer and MLP, then the final norm: one command buffer
//!   (`set_async_encode(true)`), one `synchronize()`. This is the headline. The
//!   LM head is timed separately (below) and added for the `+ lm_head` column.
//! * **per layer / per stage** — each unit encoded `REPS` times into one
//!   command buffer and divided, so the ~0.25 ms submit-and-wait floor
//!   (`docs/benchmarking.md`) is not counted once per stage. The stage table
//!   says where a layer's time goes. Summed layers are printed next to the
//!   measured forward as a cross-check.
//! * **lm_head** — `[rows, 2048] @ [2048, 248320]` into f32 is 8 GB at T = 8192,
//!   so it is timed on `LM_ROWS` rows and scaled linearly in T. It is labelled
//!   as scaled wherever it is used.
//!
//! Not counted: the embedding gather, and anything after the logits.
//!
//! Before anything is timed, one forward runs with every intermediate buffer
//! pre-filled with NaN, and every buffer a stage writes must come back finite.
//! A stage that silently wrote nothing would otherwise post the best number in
//! the table. The run aborts if the gate fails; there is no timing without it.

use std::sync::Arc;
use std::time::Instant;

use tessl::gemm::cast_f32_to_bf16_into;
use tessl::qwen35::{
    self, AttnProjLayout, AttnShape, AttnTargets, Cols, GdnChunkPhase, GdnDims, GdnParams, GdnProjLayout, GdnWorkspace,
    OutCols, StateIn,
};
use tessl::tensor::{f32_slice_to_bf16, GpuBuffer};
use tessl::{gemm, gemm_epilogue, nn, DType, Epilogue, GemmBackend, GpuRuntime, Tensor};

// Qwen3.5-2B `text_config` (models--Qwen--Qwen3.5-2B-Base config.json).
const HIDDEN: usize = 2048;
const INTER: usize = 6144;
const VOCAB: usize = 248_320;
const GDN_K_HEADS: u32 = 16;
const GDN_V_HEADS: u32 = 16;
const GDN_V_DIM: u32 = 128;
const CONV_KW: u32 = 4;
const Q_HEADS: u32 = 8;
const KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 256;
const ROTARY_DIM: u32 = 64; // head_dim * partial_rotary_factor 0.25
const ROPE_THETA: f32 = 1e7;
const EPS: f32 = 1e-6;
const LAYERS: usize = 24;
const FULL_ATTENTION_INTERVAL: usize = 4;

const BACKEND: GemmBackend = GemmBackend::TensorOps;
const REPS: usize = 8;
const WARMUP: usize = 2;
const ITERS: usize = 7;
const LM_ROWS: usize = 1024;
const DEFAULT_T: [usize; 3] = [1024, 2048, 8192];

type Res<T> = Result<T, String>;

fn is_full_attention(layer: usize) -> bool {
    layer % FULL_ATTENTION_INTERVAL == FULL_ATTENTION_INTERVAL - 1
}

fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((((s >> 32) as u32) as f64 / (u32::MAX as f64) * 2.0 - 1.0) as f32) * scale
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

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
    f32_slice_to_bf16(&fill(rows * cols, seed, (3.0 / rows as f32).sqrt()))
}

// ------------------------------------------------------------------ weights ---

struct GdnWeights {
    w_in: Tensor,
    w_out: Tensor,
    conv_w: GpuBuffer,
    a_log: GpuBuffer,
    dt_bias: GpuBuffer,
    norm_w: GpuBuffer,
}

struct AttnWeights {
    w_in: Tensor,
    w_out: Tensor,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
}

enum Mixer {
    Gdn(GdnWeights),
    Attn(AttnWeights),
}

struct Layer {
    mixer: Mixer,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
}

struct Model {
    gdn: GdnProjLayout,
    attn: AttnProjLayout,
    layers: Vec<Layer>,
    norm_w: GpuBuffer,
}

impl Model {
    fn new(rt: &Arc<GpuRuntime>) -> Res<Self> {
        let gdn = GdnProjLayout::new(GDN_K_HEADS, GDN_V_HEADS, GDN_V_DIM)?;
        let attn = AttnProjLayout::new(Q_HEADS, KV_HEADS, HEAD_DIM)?;
        let (gw, aw) = (gdn.width() as usize, attn.width() as usize);
        let gdn_out = (GDN_V_HEADS * GDN_V_DIM) as usize;
        let attn_out = (Q_HEADS * HEAD_DIM) as usize;

        let bits_gdn_in = weight_bits(HIDDEN, gw, 1);
        let bits_gdn_out = weight_bits(gdn_out, HIDDEN, 2);
        let bits_attn_in = weight_bits(HIDDEN, aw, 3);
        let bits_attn_out = weight_bits(attn_out, HIDDEN, 4);
        let bits_gate = weight_bits(HIDDEN, INTER, 5);
        let bits_up = weight_bits(HIDDEN, INTER, 6);
        let bits_down = weight_bits(INTER, HIDDEN, 7);
        let conv_w = fill(gdn.conv_dim() as usize * CONV_KW as usize, 8, 0.5);
        let vh = GDN_V_HEADS as usize;
        // Qwen's A_log sits around log(1..16) and dt_bias around -2..-7.
        let a_log: Vec<f32> = fill(vh, 9, 1.0).iter().map(|v| 1.4 + 1.3 * v).collect();
        let dt_bias: Vec<f32> = fill(vh, 10, 1.0).iter().map(|v| -4.5 + 2.5 * v).collect();

        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let mixer = if is_full_attention(l) {
                Mixer::Attn(AttnWeights {
                    w_in: weight(rt, HIDDEN, aw, &bits_attn_in)?,
                    w_out: weight(rt, attn_out, HIDDEN, &bits_attn_out)?,
                    q_norm: buf(rt, &fill(HEAD_DIM as usize, 11, 0.1))?,
                    k_norm: buf(rt, &fill(HEAD_DIM as usize, 12, 0.1))?,
                })
            } else {
                Mixer::Gdn(GdnWeights {
                    w_in: weight(rt, HIDDEN, gw, &bits_gdn_in)?,
                    w_out: weight(rt, gdn_out, HIDDEN, &bits_gdn_out)?,
                    conv_w: buf(rt, &conv_w)?,
                    a_log: buf(rt, &a_log)?,
                    dt_bias: buf(rt, &dt_bias)?,
                    norm_w: buf(
                        rt,
                        &fill(GDN_V_DIM as usize, 13, 0.1)
                            .iter()
                            .map(|v| 1.0 + v)
                            .collect::<Vec<_>>(),
                    )?,
                })
            };
            layers.push(Layer {
                mixer,
                w_gate: weight(rt, HIDDEN, INTER, &bits_gate)?,
                w_up: weight(rt, HIDDEN, INTER, &bits_up)?,
                w_down: weight(rt, INTER, HIDDEN, &bits_down)?,
            });
        }
        let norm_w = buf(rt, &vec![1.0f32; HIDDEN])?;
        rt.synchronize()?;
        Ok(Self {
            gdn,
            attn,
            layers,
            norm_w,
        })
    }
}

// -------------------------------------------------------------- activations ---

/// Every intermediate of one forward at `t` tokens, shared by all layers (the
/// runtime's per-dispatch barriers order the reuse).
struct Acts {
    t: usize,
    resid: Tensor,
    xb: Tensor,
    // GDN
    g_proj: Tensor,
    g_qkv: GpuBuffer,
    g_o: GpuBuffer,
    g_y: Tensor,
    g_ws: GdnWorkspace,
    g_dims: GdnDims,
    // attention
    a_proj: Tensor,
    a_q: GpuBuffer,
    a_kc: GpuBuffer,
    a_vc: GpuBuffer,
    a_o: GpuBuffer,
    a_y: Tensor,
    tkv: GpuBuffer,
    zero_pos: GpuBuffer,
    // MLP
    m_gate: Tensor,
    m_up: Tensor,
    m_mid: Tensor,
    m_midb: Tensor,
}

impl Acts {
    fn new(rt: &Arc<GpuRuntime>, m: &Model, t: usize) -> Res<Self> {
        let g_dims = m.gdn.dims(1, t as u32);
        let kv = t * (KV_HEADS * HEAD_DIM) as usize;
        let qd = (Q_HEADS * HEAD_DIM) as usize;
        let acts = Self {
            t,
            resid: rt.alloc_tensor_f32(&[t, HIDDEN])?,
            xb: rt.alloc_tensor_bf16(&[t, HIDDEN])?,
            g_proj: rt.alloc_tensor_f32(&[t, m.gdn.width() as usize])?,
            g_qkv: rt.alloc_buffer(t * m.gdn.conv_dim() as usize * 4)?,
            g_o: rt.alloc_buffer(t * m.gdn.value_dim() as usize * 4)?,
            g_y: rt.alloc_tensor_bf16(&[t, m.gdn.value_dim() as usize])?,
            g_ws: GdnWorkspace::new(rt, &g_dims)?.with_scan_slice(*GDN_SCAN.get().expect("set in main")),
            g_dims,
            a_proj: rt.alloc_tensor_f32(&[t, m.attn.width() as usize])?,
            a_q: rt.alloc_buffer(t * qd * 4)?,
            a_kc: rt.alloc_buffer(kv * 4)?,
            a_vc: rt.alloc_buffer(kv * 4)?,
            a_o: rt.alloc_buffer(t * qd * 4)?,
            a_y: rt.alloc_tensor_bf16(&[t, qd])?,
            tkv: buf_u32(rt, &[t as u32])?,
            zero_pos: buf_u32(rt, &[0])?,
            m_gate: rt.alloc_tensor_f32(&[t, INTER])?,
            m_up: rt.alloc_tensor_f32(&[t, INTER])?,
            m_mid: rt.alloc_tensor_f32(&[t, INTER])?,
            m_midb: rt.alloc_tensor_bf16(&[t, INTER])?,
        };
        for tensor in [&acts.g_proj, &acts.a_proj] {
            if tensor.byte_offset() != 0 {
                return Err("the qwen35 kernels address the projection from its buffer's start".into());
            }
        }
        acts.reset_resid()?;
        Ok(acts)
    }

    fn reset_resid(&self) -> Res<()> {
        self.resid.buffer.write_f32(&fill(self.t * HIDDEN, 99, 1.0));
        Ok(())
    }

    /// Fill every intermediate with NaN, so the gate can tell written from not.
    fn poison(&self) {
        // The write_* calls check the whole mapping, so size by the allocation.
        let nan = |b: &GpuBuffer| b.write_f32(&vec![f32::NAN; b.nbytes() / 4]);
        let nan_bf16 = |b: &GpuBuffer| b.write_bf16_bits(&vec![0x7fc0u16; b.nbytes() / 2]);
        for b in [&self.xb, &self.g_y, &self.a_y, &self.m_midb] {
            nan_bf16(&b.buffer);
        }
        for b in [&self.g_proj, &self.a_proj, &self.m_gate, &self.m_up, &self.m_mid] {
            nan(&b.buffer);
        }
        for b in [&self.g_qkv, &self.g_o, &self.a_q, &self.a_kc, &self.a_vc, &self.a_o] {
            nan(b);
        }
    }
}

// ------------------------------------------------------------------- stages ---

fn input_norm(rt: &Arc<GpuRuntime>, m: &Model, a: &Acts) -> Res<()> {
    nn::rms_norm_bf16(
        rt,
        &a.resid.buffer,
        &m.norm_w,
        &a.xb.buffer,
        a.t as u32,
        HIDDEN as u32,
        EPS,
    )
}

fn residual_add() -> Epilogue<'static> {
    Epilogue {
        beta: 1.0,
        ..Epilogue::default()
    }
}

/// The GDN mixer's stages, in order. `input_norm` precedes them in a layer.
/// `gdn_chunk_forward` is its two dispatches, timed apart; run in order they are
/// exactly `gdn_chunk_forward`.
const GDN_STAGES: [&str; 6] = [
    "in-proj GEMM",
    "conv1d_silu",
    "gdn chunk prep",
    "gdn chunk scan",
    "gated_rms_norm",
    "out-proj + resid",
];

fn gdn_stage(rt: &Arc<GpuRuntime>, m: &Model, w: &GdnWeights, a: &Acts, s: usize) -> Res<()> {
    let proj = &a.g_proj.buffer;
    let vd = m.gdn.value_dim();
    match s {
        0 => qwen35::fused_projection(&a.xb, &w.w_in, &a.g_proj, BACKEND),
        1 => qwen35::conv1d_silu(
            rt,
            Cols::dense(proj, m.gdn.width()),
            &w.conv_w,
            CONV_KW,
            StateIn::Zero,
            &a.g_qkv,
            None,
            1,
            a.t as u32,
            m.gdn.conv_dim(),
        ),
        2 | 3 => qwen35::gdn_chunk_phase(
            rt,
            &a.g_dims,
            &m.gdn.conv_qkv(&a.g_qkv),
            &m.gdn.gates(proj),
            &GdnParams {
                a_log: &w.a_log,
                dt_bias: &w.dt_bias,
            },
            StateIn::Zero,
            &a.g_ws,
            Cols::dense(&a.g_o, vd),
            None,
            if s == 2 {
                GdnChunkPhase::Prep
            } else {
                GdnChunkPhase::Scan
            },
        ),
        4 => qwen35::gated_rms_norm(
            rt,
            Cols::dense(&a.g_o, vd),
            m.gdn.z(proj),
            &w.norm_w,
            OutCols {
                cols: Cols::dense(&a.g_y.buffer, vd),
                dtype: DType::BF16,
            },
            a.t as u32,
            GDN_V_HEADS,
            GDN_V_DIM,
            EPS,
        ),
        5 => qwen35::project_residual(&a.g_y, &w.w_out, &a.resid, BACKEND),
        _ => unreachable!(),
    }
}

const ATTN_STAGES: [&str; 5] = [
    "qkv+gate GEMM",
    "qk_norm_rope",
    "attention",
    "output gate",
    "o-proj + resid",
];

/// `--gdn-scan16`: the chunked GDN rule's scan in 16-column slices.
static GDN_SCAN: std::sync::OnceLock<qwen35::GdnScanSlice> = std::sync::OnceLock::new();

/// Which attention kernel the attention stage times.
#[derive(Clone, Copy, Debug)]
enum AttnChoice {
    /// `--attn-rows`: the scalar `nn::flash_attn_rows`.
    Rows,
    /// `qwen35::attn_prefill_with_tile` (default: `ATTN_PREFILL_TILE`).
    Tiled(qwen35::AttnTile),
}

static ATTN: std::sync::OnceLock<AttnChoice> = std::sync::OnceLock::new();

fn attn_stage(rt: &Arc<GpuRuntime>, m: &Model, w: &AttnWeights, a: &Acts, s: usize) -> Res<()> {
    let pc = Cols::dense(&a.a_proj.buffer, m.attn.width());
    match s {
        0 => qwen35::fused_projection(&a.xb, &w.w_in, &a.a_proj, BACKEND),
        1 => qwen35::attn_qk_norm_rope(
            rt,
            &AttnShape {
                batch: 1,
                seq: a.t as u32,
                q_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                rotary_dim: ROTARY_DIM,
            },
            pc,
            &w.q_norm,
            &w.k_norm,
            &AttnTargets {
                q_out: &a.a_q,
                k_cache: &a.a_kc,
                v_cache: &a.a_vc,
            },
            0,
            ROPE_THETA,
            EPS,
        ),
        2 => {
            let dims = nn::AttnDims {
                batch: 1,
                tq: a.t as u32,
                heads: Q_HEADS,
                heads_kv: KV_HEADS,
                window: 0,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            };
            let (q, k, v, o, pos) = (&a.a_q, &a.a_kc, &a.a_vc, &a.a_o, &a.zero_pos);
            match *ATTN.get().expect("set in main") {
                AttnChoice::Rows => nn::flash_attn_rows(rt, q, k, v, o, &a.tkv, pos, pos, dims, HEAD_DIM, false),
                AttnChoice::Tiled(tile) => {
                    qwen35::attn_prefill_with_tile(rt, q, k, v, o, &a.tkv, pos, pos, dims, false, tile)
                }
            }
        }
        3 => qwen35::attn_output_gate(
            rt,
            &a.a_o,
            pc,
            OutCols {
                cols: Cols::dense(&a.a_y.buffer, Q_HEADS * HEAD_DIM),
                dtype: DType::BF16,
            },
            a.t as u32,
            Q_HEADS,
            HEAD_DIM,
        ),
        4 => gemm_epilogue(&a.a_y, &w.w_out, &a.resid, BACKEND, residual_add()),
        _ => unreachable!(),
    }
}

/// The MLP block's stages, post-attention norm first. SwiGLU is
/// `qwen35::swiglu` straight to bf16 for the down GEMM, or with
/// `--mlp-unfused` the generic `nn::mlp_silu` into f32 plus a cast pass, the
/// path it replaced; either way it is one row, so the two compare directly.
const MLP_STAGES: [&str; 5] = ["post-norm", "gate GEMM", "up GEMM", "swiglu -> bf16", "down + resid"];

/// `--mlp-unfused`: time `nn::mlp_silu` + cast instead of `qwen35::swiglu`.
static MLP_UNFUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn mlp_stage(rt: &Arc<GpuRuntime>, m: &Model, l: &Layer, a: &Acts, s: usize) -> Res<()> {
    match s {
        0 => input_norm(rt, m, a),
        1 => gemm(&a.xb, &l.w_gate, &a.m_gate, BACKEND),
        2 => gemm(&a.xb, &l.w_up, &a.m_up, BACKEND),
        3 if *MLP_UNFUSED.get().expect("set in main") => {
            nn::mlp_silu(
                rt,
                &a.m_gate.buffer,
                &a.m_up.buffer,
                &a.m_mid.buffer,
                (a.t * INTER) as u32,
            )?;
            cast_f32_to_bf16_into(&a.m_mid, &a.m_midb)
        }
        3 => qwen35::swiglu(
            rt,
            Cols::dense(&a.m_gate.buffer, INTER as u32),
            Cols::dense(&a.m_up.buffer, INTER as u32),
            OutCols {
                cols: Cols::dense(&a.m_midb.buffer, INTER as u32),
                dtype: DType::BF16,
            },
            a.t as u32,
            INTER as u32,
        ),
        4 => gemm_epilogue(&a.m_midb, &l.w_down, &a.resid, BACKEND, residual_add()),
        _ => unreachable!(),
    }
}

fn mixer(rt: &Arc<GpuRuntime>, m: &Model, l: &Layer, a: &Acts) -> Res<()> {
    input_norm(rt, m, a)?;
    match &l.mixer {
        Mixer::Gdn(w) => (0..GDN_STAGES.len()).try_for_each(|s| gdn_stage(rt, m, w, a, s)),
        Mixer::Attn(w) => (0..ATTN_STAGES.len()).try_for_each(|s| attn_stage(rt, m, w, a, s)),
    }
}

fn mlp(rt: &Arc<GpuRuntime>, m: &Model, l: &Layer, a: &Acts) -> Res<()> {
    (0..MLP_STAGES.len()).try_for_each(|s| mlp_stage(rt, m, l, a, s))
}

fn forward(rt: &Arc<GpuRuntime>, m: &Model, a: &Acts) -> Res<()> {
    for l in &m.layers {
        mixer(rt, m, l, a)?;
        mlp(rt, m, l, a)?;
    }
    input_norm(rt, m, a)
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

fn finite_bf16(name: &str, t: &Tensor) -> Res<()> {
    let n = t.numel();
    let bits = t.buffer.contents_u16();
    match bits[..n].iter().position(|&h| h & 0x7f80 == 0x7f80) {
        Some(i) => Err(format!(
            "{name}: bf16 element {i} of {n} is non-finite (0x{:04x})",
            bits[i]
        )),
        None => Ok(()),
    }
}

/// One NaN-poisoned forward; every buffer the last GDN, attention and MLP stages
/// wrote must be finite, and the residual must have moved.
fn plausibility_gate(rt: &Arc<GpuRuntime>, m: &Model, a: &Acts) -> Res<()> {
    rt.synchronize()?;
    a.reset_resid()?;
    a.poison();
    let before = a.resid.buffer.read_f32();
    forward(rt, m, a)?;
    rt.synchronize()?;
    let t = a.t;
    let kv = t * (KV_HEADS * HEAD_DIM) as usize;
    let qd = t * (Q_HEADS * HEAD_DIM) as usize;
    finite_f32("residual", &a.resid.buffer, t * HIDDEN)?;
    finite_bf16("final norm", &a.xb)?;
    finite_f32("gdn in-proj", &a.g_proj.buffer, a.g_proj.numel())?;
    finite_f32("gdn conv", &a.g_qkv, t * m.gdn.conv_dim() as usize)?;
    finite_f32("gdn chunk out", &a.g_o, t * m.gdn.value_dim() as usize)?;
    finite_bf16("gdn gated norm", &a.g_y)?;
    finite_f32("attn proj", &a.a_proj.buffer, a.a_proj.numel())?;
    finite_f32("attn q", &a.a_q, qd)?;
    finite_f32("attn k cache", &a.a_kc, kv)?;
    finite_f32("attn v cache", &a.a_vc, kv)?;
    finite_f32("attention out", &a.a_o, qd)?;
    finite_bf16("attn gated out", &a.a_y)?;
    finite_f32("mlp gate", &a.m_gate.buffer, a.m_gate.numel())?;
    finite_f32("mlp up", &a.m_up.buffer, a.m_up.numel())?;
    if *MLP_UNFUSED.get().expect("set in main") {
        finite_f32("mlp silu", &a.m_mid.buffer, a.m_mid.numel())?;
    }
    finite_bf16("mlp swiglu", &a.m_midb)?;
    let after = a.resid.buffer.read_f32();
    let moved = before.iter().zip(&after).filter(|(x, y)| x != y).count();
    if moved < t * HIDDEN / 2 {
        return Err(format!(
            "residual: only {moved} of {} elements changed over 24 layers",
            t * HIDDEN
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------- timing ---

/// Median milliseconds of one `f`, with `reps` of it per command buffer.
fn time_ms(rt: &Arc<GpuRuntime>, reps: usize, mut f: impl FnMut() -> Res<()>) -> Res<f64> {
    rt.synchronize()?;
    for _ in 0..WARMUP {
        for _ in 0..reps {
            f()?;
        }
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        for _ in 0..reps {
            f()?;
        }
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
    }
    Ok(median(samples))
}

struct Timed {
    t: usize,
    gdn_layer: f64,
    attn_layer: f64,
    mlp: f64,
    forward: f64,
    dispatches: usize,
    stages: Vec<(String, String, f64)>,
}

fn run_t(rt: &Arc<GpuRuntime>, m: &Model, a: &Acts) -> Res<Timed> {
    let gdn_l = m.layers.iter().find(|l| matches!(l.mixer, Mixer::Gdn(_))).unwrap();
    let attn_l = m.layers.iter().find(|l| matches!(l.mixer, Mixer::Attn(_))).unwrap();
    let (Mixer::Gdn(gw), Mixer::Attn(aw)) = (&gdn_l.mixer, &attn_l.mixer) else {
        unreachable!()
    };

    rt.synchronize()?;
    rt.take_dispatch_count();
    forward(rt, m, a)?;
    rt.synchronize()?;
    let dispatches = rt.take_dispatch_count();

    let forward_ms = time_ms(rt, 1, || forward(rt, m, a))?;
    let gdn_layer = time_ms(rt, REPS, || mixer(rt, m, gdn_l, a))?;
    let attn_layer = time_ms(rt, REPS, || mixer(rt, m, attn_l, a))?;
    let mlp_ms = time_ms(rt, REPS, || mlp(rt, m, gdn_l, a))?;

    let mut stages = Vec::new();
    stages.push((
        "gdn".to_string(),
        "input norm".to_string(),
        time_ms(rt, REPS, || input_norm(rt, m, a))?,
    ));
    for (s, name) in GDN_STAGES.iter().enumerate() {
        let ms = time_ms(rt, REPS, || gdn_stage(rt, m, gw, a, s))?;
        stages.push(("gdn".into(), (*name).into(), ms));
    }
    for (s, name) in ATTN_STAGES.iter().enumerate() {
        // flash attention reads what qk_norm_rope cached; its inputs stay valid
        // across repeats because every stage rewrites the same slots.
        let ms = time_ms(rt, REPS, || attn_stage(rt, m, aw, a, s))?;
        stages.push(("attn".into(), (*name).into(), ms));
    }
    for (s, name) in MLP_STAGES.iter().enumerate() {
        let ms = time_ms(rt, REPS, || mlp_stage(rt, m, gdn_l, a, s))?;
        stages.push(("mlp".into(), (*name).into(), ms));
    }
    Ok(Timed {
        t: a.t,
        gdn_layer,
        attn_layer,
        mlp: mlp_ms,
        forward: forward_ms,
        dispatches,
        stages,
    })
}

/// Milliseconds per row of the full-vocabulary LM head, from `LM_ROWS` rows.
fn lm_head_ms_per_row(rt: &Arc<GpuRuntime>) -> Res<f64> {
    let w = weight(rt, HIDDEN, VOCAB, &weight_bits(HIDDEN, VOCAB, 20))?;
    let x = rt.alloc_tensor_bf16(&[LM_ROWS, HIDDEN])?;
    x.buffer
        .write_bf16_bits(&f32_slice_to_bf16(&fill(LM_ROWS * HIDDEN, 21, 1.0)));
    let logits = rt.alloc_tensor_f32(&[LM_ROWS, VOCAB])?;
    logits.buffer.write_f32(&vec![f32::NAN; LM_ROWS * VOCAB]);
    gemm(&x, &w, &logits, BACKEND)?;
    rt.synchronize()?;
    finite_f32("lm_head logits", &logits.buffer, LM_ROWS * VOCAB)?;
    Ok(time_ms(rt, 1, || gemm(&x, &w, &logits, BACKEND))? / LM_ROWS as f64)
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
    Tiled,
}

impl PairKernel {
    fn name(self) -> &'static str {
        match self {
            PairKernel::Rows => "flash_attn_rows",
            PairKernel::Tiled => "attn_prefill",
        }
    }
}

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
        PairKernel::Tiled => qwen35::attn_prefill(rt, q, k, v, o, tkv, pos, pos, dims, false),
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
        let q = buf(rt, &fill(qd, 11, scale))?;
        let k = buf(rt, &fill(kv, 12, scale))?;
        let v = buf(rt, &fill(kv, 13, scale))?;
        let o = rt.alloc_buffer(qd * 4)?;
        let tkv = buf_u32(rt, &[t as u32])?;
        let pos = buf_u32(rt, &[0])?;

        let mut checksums = Vec::new();
        for kernel in [PairKernel::Rows, PairKernel::Tiled] {
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
                [PairKernel::Rows, PairKernel::Tiled]
            } else {
                [PairKernel::Tiled, PairKernel::Rows]
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
                        PairKernel::Tiled => tiled.push(ms),
                    }
                }
            }
        }

        for (name, samples) in [("flash_attn_rows", &rows), ("attn_prefill", &tiled)] {
            let med = median(samples.clone());
            let lo = samples.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "PAIRED t={t} kernel={name} n={} reps={reps} median_ms={med:.4} min_ms={lo:.4} max_ms={hi:.4} buffer_median_ms={:.2}",
                samples.len(),
                med * reps as f64
            );
        }
        let ratio = median(tiled.clone()) / median(rows.clone());
        println!("PAIRED t={t} ratio_tiled_over_rows={ratio:.4}");
    }
    Ok(())
}

// --------------------------------------------------------------------- main ---

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut check_only = false;
    let mut paired_attn = false;
    let mut attn = AttnChoice::Tiled(qwen35::ATTN_PREFILL_TILE);
    let mut mlp_unfused = false;
    let mut gdn_scan = qwen35::GdnScanSlice::Cols32;
    let mut ts = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--check-only" {
            check_only = true;
        } else if arg == "--paired-attn" {
            paired_attn = true;
        } else if arg == "--gdn-scan16" {
            gdn_scan = qwen35::GdnScanSlice::Cols16;
        } else if arg == "--mlp-unfused" {
            mlp_unfused = true;
        } else if arg == "--attn-rows" {
            attn = AttnChoice::Rows;
        } else if let Some(label) = arg.strip_prefix("--attn-tile=") {
            let tile = qwen35::AttnTile::ALL
                .into_iter()
                .find(|t| t.label() == label)
                .ok_or_else(|| {
                    let known: Vec<String> = qwen35::AttnTile::ALL.iter().map(|t| t.label()).collect();
                    format!("unknown --attn-tile {label:?}; one of {}", known.join(", "))
                })?;
            attn = AttnChoice::Tiled(tile);
        } else {
            let t: usize = arg.parse().map_err(|_| {
                format!(
                    "expected a token count, --check-only, --paired-attn, --attn-rows, --attn-tile=LABEL, \
                     --mlp-unfused or --gdn-scan16, got {arg:?}"
                )
            })?;
            if t == 0 {
                return Err("T must be positive".into());
            }
            ts.push(t);
        }
    }
    if paired_attn && check_only {
        return Err("--paired-attn times both kernels; it does not combine with --check-only".into());
    }
    if ts.is_empty() {
        ts = if paired_attn {
            vec![200, 8192]
        } else {
            DEFAULT_T.to_vec()
        };
    }

    let rt = GpuRuntime::new()?;
    if !rt.has_tensorops() {
        return Err("bf16 GEMMs need the TensorOps backend, which this device lacks".into());
    }
    println!("device: {}", rt.device_name());
    if paired_attn {
        rt.set_async_encode(true)?;
        run_paired_attn(&rt, &ts)?;
        rt.set_async_encode(false)?;
        return Ok(());
    }
    println!(
        "Qwen3.5-2B shapes: hidden {HIDDEN}, GDN {GDN_K_HEADS}k/{GDN_V_HEADS}v heads x {GDN_V_DIM}, \
         attn {Q_HEADS}q/{KV_HEADS}kv x {HEAD_DIM} (rotary {ROTARY_DIM}), MLP {INTER}, vocab {VOCAB}"
    );
    let n_attn = (0..LAYERS).filter(|&l| is_full_attention(l)).count();
    let n_gdn = LAYERS - n_attn;
    println!("layers: {n_gdn} GDN + {n_attn} attention, batch 1, bf16 weights distinct per layer");
    GDN_SCAN.set(gdn_scan).map_err(|_| "GDN scan slice chosen twice")?;
    println!("gdn scan: {gdn_scan:?}");
    MLP_UNFUSED.set(mlp_unfused).map_err(|_| "MLP path chosen twice")?;
    println!(
        "swiglu: {}",
        if mlp_unfused {
            "nn::mlp_silu (f32) + cast to bf16"
        } else {
            "qwen35::swiglu (bf16 out)"
        }
    );
    ATTN.set(attn).map_err(|_| "attention kernel chosen twice")?;
    match attn {
        AttnChoice::Rows => println!("attention kernel: nn::flash_attn_rows (scalar f32)"),
        AttnChoice::Tiled(tile) => println!(
            "attention kernel: qwen35::attn_prefill (TensorOps f32), tile {}",
            tile.label()
        ),
    }

    let t0 = Instant::now();
    let model = Model::new(&rt)?;
    println!("weights uploaded in {:.1} s", t0.elapsed().as_secs_f64());

    // Gate under the same encoding the timing uses: every layer in one command
    // buffer, sharing activations through the runtime's barriers.
    rt.set_async_encode(true)?;
    for &t in &ts {
        let acts = Acts::new(&rt, &model, t)?;
        plausibility_gate(&rt, &model, &acts).map_err(|e| format!("T={t}: plausibility gate failed: {e}"))?;
        println!(
            "T={t}: plausibility gate passed (every stage output finite, residual moved; \
             GDN workspace {:.0} MB)",
            GdnWorkspace::bytes_for(&acts.g_dims)? as f64 / 1e6
        );
    }
    if check_only {
        rt.set_async_encode(false)?;
        println!("--check-only: nothing timed");
        return Ok(());
    }

    let lm_row = lm_head_ms_per_row(&rt)?;
    let mut results = Vec::new();
    for &t in &ts {
        let acts = Acts::new(&rt, &model, t)?;
        results.push(run_t(&rt, &model, &acts)?);
    }
    rt.set_async_encode(false)?;

    println!();
    println!(
        "per layer (ms, median of {ITERS}, {REPS} per command buffer). \
         GDN / attention layer = input norm + mixer; MLP = post-norm + SwiGLU"
    );
    println!(
        "{:>6} {:>9} {:>9} {:>9} | {:>11} {:>11} | {:>10} {:>12} {:>9}",
        "T", "GDN", "attn", "MLP", "sum layers", "forward", "tok/s", "+lm_head*", "launches"
    );
    for r in &results {
        let summed = n_gdn as f64 * r.gdn_layer + n_attn as f64 * r.attn_layer + LAYERS as f64 * r.mlp;
        let lm = lm_row * r.t as f64;
        println!(
            "{:>6} {:>9.3} {:>9.3} {:>9.3} | {:>11.2} {:>11.2} | {:>10.0} {:>12.0} {:>9}",
            r.t,
            r.gdn_layer,
            r.attn_layer,
            r.mlp,
            summed,
            r.forward,
            r.t as f64 / r.forward * 1e3,
            r.t as f64 / (r.forward + lm) * 1e3,
            r.dispatches,
        );
    }
    println!(
        "forward = all {LAYERS} layers + final norm in one command buffer (measured, not summed). \
         * lm_head scaled from {LM_ROWS} rows: {:.4} ms/row",
        lm_row
    );

    println!();
    println!("per stage (ms)");
    print!("{:<6} {:<18}", "", "stage");
    for r in &results {
        print!(" {:>9}", format!("T={}", r.t));
    }
    println!();
    for i in 0..results[0].stages.len() {
        let (group, name, _) = &results[0].stages[i];
        print!("{group:<6} {name:<18}");
        for r in &results {
            print!(" {:>9.3}", r.stages[i].2);
        }
        println!();
    }
    Ok(())
}
