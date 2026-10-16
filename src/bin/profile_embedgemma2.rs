//! Where an EmbedGemma 2 encode's GPU time and device memory go, per stage.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   PROFILE_MODE=stages cargo run --release --bin profile_embedgemma2
//! ```
//!
//! `PROFILE_MODE` picks one of:
//!
//! * `stages` (default): a replica of `EmbedGemma2Model::forward` built from the
//!   library's public kernels, loaded from the real checkpoint (all 24 layers, so
//!   the weights cycle through the caches as they do in an encode). The replica
//!   runs in async mode, `reps` forwards queued per host wait, and each of its
//!   stages is a *class* that a bit mask switches on or off. Per workload it
//!   times, interleaved and min-of-`PROFILE_ROUNDS`:
//!   * `full`: every class on. Its output is compared with `encode`'s on the same
//!     tokens (a replica that does not reproduce the library is not measuring it);
//!   * `only:<class>`: one class on, the rest skipped: that stage's cost alone,
//!     with the real shapes, the real weights of all 24 layers and the same
//!     dependency barriers between dispatches;
//!   * `skip:<class>`: the full forward without it. `full - skip` is the stage's
//!     marginal cost in the pipelined forward (what removing or fusing it
//!     saves, if the rest were unchanged);
//!   * `none`: no dispatch, one empty wait (the harness floor);
//!   * `encode`: `EmbedGemma2Model::encode` itself, upload and readback included.
//!
//!   Each sample also carries the host time to *encode* the queued forwards
//!   (before the wait): when that is close to the total, the host, not the GPU,
//!   is the limit. A skipped stage leaves stale data in its output buffers; GPU
//!   time does not depend on values, and every buffer is zero-initialized so a
//!   skipped stage can never feed a NaN downstream.
//! * `gemm`: the forward's six GEMM shapes (and the per-layer-input projection)
//!   in isolation at 16 / 2048 / 8192 rows, with a distinct weight per layer
//!   cycled so the weights are not cache-resident, in exact f32 and with bf16
//!   operands (`gemm_bf16`, f32 accumulation): the measured upper bound of a bf16
//!   path, plus measured machine ceilings (copy bandwidth, large-GEMM rate).
//! * `memory`: one fresh process's device-memory trace for one workload:
//!   `PROFILE_STEPS` is a comma list of `E` (encode), `S` (synchronize), `A` (allocate the
//!   activation set `acts()` makes, by the same formulas), `O` (the same for
//!   2a7cb8b's), `F` (free what `A`/`O` allocated); `PROFILE_PLE_BYTES` overrides
//!   `set_ple_run_bytes`; `PROFILE_OPERANDS=f32|bf16` (default `f32`) picks the
//!   model's `GemmOperands`. Prints one JSON object with the allocated bytes above
//!   the loaded model and the device peak after each step.
//!   `PROFILE_PRELUDE` (workloads) encodes each twice first, as
//!   `bench_embedgemma2` does before the workload whose peak it reports.
//! * `memory-sweep`: runs `memory` in a fresh child process per configuration.
//! * `ledger`: one synchronous replica forward with `synchronize` and a peak
//!   reset around every stage: the device bytes each stage allocated fresh.
//! * `memory-table`: `memory` for each workload under both `GemmOperands`:
//!   one `encode`, back-to-back `encode`s, `encode`s with a `synchronize`
//!   between (see `mode_memory_table`).
//! * `layout`: the activation set itemized, HEAD against 2a7cb8b, with the
//!   bytes the pool charges for each. No GPU.
//! * `alloc`: what allocating and dropping `encode`'s activation set costs
//!   per call, with no forward.
//! * `ple`: `encode` timed at several `set_ple_run_bytes`.
//! * `hash`: FNV-1a hashes of `encode`'s embeddings per workload under both
//!   `GemmOperands` (see `mode_hash`), to diff two builds bit for bit.
//!
//! `stages` also times `full, fresh ... activations per call` (the forward on
//! a new activation set per call, zeroed and unzeroed, as `encode` makes it),
//! and `PROFILE_SKIP=0` leaves out the `skip:` variants. Before every sample
//! the harness waits (at most `PROFILE_WAIT_S` seconds in all, default 1800)
//! until no other test or bench binary of a tessl checkout is running, retakes
//! a sample such a process appeared during, and flags what it could not make
//! clean; the statistics use the clean samples. Processes it cannot see
//! (display, browsers, other repositories) remain, and the `ioreg` GPU
//! utilization at the start and end of each workload records them.
//!
//! `BENCH_WORKLOADS` (default `1x16,64x32,32x256,8x1024,2x4096`) lists `BxT`
//! uniform workloads (the grammar of `bench_embedgemma2`, uniform terms only);
//! `PROFILE_JSON` names the output file. Nothing here changes library code.

mod common;

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tessl::embedgemma2::{
    l2_normalize_rows, segment_mean_rows, EmbedGemma2Config, EmbedGemma2Model, EncoderAttnDims, LayerSpec,
};
use tessl::gemm::{gemm, gemm_bf16, gemm_nt_f32, GemmBackend, GemmOperands};
use tessl::nn::{rms_norm_f32, rms_norm_residual_add_f32, scale_f32_inplace};
use tessl::qwen35::{self, AttnShape, AttnTargets, Cols, GatedAct, LmHead, OutCols, QkvColumns};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{gpu_copy, DType, GpuBuffer, Tensor};
use tessl::{BufferKind, GpuRuntime};

use common::{env_usize, round_order};

const BACKEND: GemmBackend = GemmBackend::TensorOps;
const MIB: f64 = (1u64 << 20) as f64;
const DEFAULT_WORKLOADS: &str = "1x16,64x32,32x256,8x1024,2x4096";
const PREFIX: &str = "language_model.";

// ---------------------------------------------------------------------------
// Stage classes
// ---------------------------------------------------------------------------

/// One switchable stage of the forward. `forward` runs a class only when its bit
/// of the mask is set.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum C {
    Gather,
    RmsNorm,
    QkvSliding,
    QkvGlobal,
    QkNormRope,
    AttnSliding,
    AttnGlobal,
    OSliding,
    OGlobal,
    NormResidAttn,
    GateUp,
    GeluMlp,
    Down,
    NormResidFfn,
    PleProj,
    PleScaleNorm,
    PleGate,
    PleGelu,
    PleOut,
    NormResidPle,
    Tail,
}

const NAMES: [&str; 21] = [
    "gather+copy",
    "rms_norm (input, pre-ffn)",
    "qkv GEMM sliding",
    "qkv GEMM global",
    "qk-norm+rope+kv store",
    "attention sliding (hd256)",
    "attention global (hd512)",
    "o GEMM sliding",
    "o GEMM global",
    "norm+residual (post-attn)",
    "gate|up GEMM",
    "gelu*up (mlp)",
    "down GEMM",
    "norm+residual (post-ffn)",
    "PLE projection GEMM (NT)",
    "PLE scale+norm",
    "ple_gate GEMM",
    "ple gelu*input",
    "ple_out GEMM",
    "norm+residual*scalar (post-ple)",
    "tail (final norm, pool, proj, l2)",
];
const N_CLASSES: usize = NAMES.len();
const ALL: u32 = (1u32 << N_CLASSES) - 1;

fn bit(c: C) -> u32 {
    1u32 << (c as usize)
}

// ---------------------------------------------------------------------------
// Weights (the real checkpoint, packed as `Loader::linear` packs them)
// ---------------------------------------------------------------------------

struct LayerW {
    spec: LayerSpec,
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
    ple_gate: Tensor,
    ple_out: Tensor,
    post_ple_norm: GpuBuffer,
    layer_scalar: f32,
}

struct Weights {
    embed: Tensor,
    ple_w: Tensor,
    ple_norm: GpuBuffer,
    final_norm: GpuBuffer,
    projection: Tensor,
    layers: Vec<LayerW>,
}

fn norm_buf(rt: &Arc<GpuRuntime>, st: &SafeTensors, name: &str, dim: usize) -> Result<GpuBuffer, String> {
    let (shape, data) = st.read_f32(&format!("{PREFIX}{name}"))?;
    if shape != [dim] {
        return Err(format!("{name}: shape {shape:?}, expected [{dim}]"));
    }
    let b = rt.alloc_buffer_hot(dim.max(1) * 4)?;
    b.try_write_f32(&data)?;
    Ok(b)
}

/// `nn.Linear` weights `[out_i, in]` packed side by side into `[in, sum(out_i)]`.
fn pack_linear(
    rt: &Arc<GpuRuntime>,
    st: &SafeTensors,
    parts: &[(String, usize)],
    inf: usize,
) -> Result<Tensor, String> {
    let total: usize = parts.iter().map(|p| p.1).sum();
    let t = rt.alloc_tensor_f32_hot(&[inf, total])?;
    let mut dst = t.buffer.try_contents_f32()?;
    let mut col0 = 0;
    for (name, out) in parts {
        let (shape, w) = st.read_f32(&format!("{PREFIX}{name}"))?;
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
    fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, cfg: &EmbedGemma2Config) -> Result<Self, String> {
        let (h, ple, inter, vocab) = (
            cfg.hidden as usize,
            cfg.ple_dim as usize,
            cfg.intermediate as usize,
            cfg.vocab as usize,
        );
        let n_layers = cfg.layers.len();
        let embed = rt.alloc_tensor_bf16_hot(&[vocab, h])?;
        let (shape, bits) = st.read_bf16_bits(&format!("{PREFIX}embed_tokens.weight"))?;
        if shape != [vocab, h] {
            return Err(format!("embed_tokens: shape {shape:?}"));
        }
        embed.buffer.try_contents_u16()?.copy_from_slice(&bits);
        drop(bits);
        let ple_w = rt.alloc_tensor_f32_hot(&[n_layers * ple, h])?;
        let (shape, data) = st.read_f32(&format!("{PREFIX}ple.per_layer_model_projection.weight"))?;
        if shape != [n_layers * ple, h] {
            return Err(format!("ple projection: shape {shape:?}"));
        }
        ple_w.buffer.try_contents_f32()?.copy_from_slice(&data);
        drop(data);
        let ple_norm = norm_buf(rt, st, "ple.per_layer_projection_norm.weight", ple)?;
        let final_norm = norm_buf(rt, st, "norm.weight", h)?;
        let projection = pack_linear(
            rt,
            st,
            &[("embedding_projection.weight".to_string(), cfg.embedding_dim as usize)],
            h,
        )?;
        let mut layers = Vec::with_capacity(n_layers);
        for (i, &spec) in cfg.layers.iter().enumerate() {
            let p = |s: &str| format!("layers.{i}.{s}");
            let (d, q, kv) = (spec.head_dim as usize, cfg.q_heads as usize, spec.kv_heads as usize);
            let scalar = st.read_f32(&format!("{PREFIX}{}", p("layer_scalar")))?.1;
            if scalar.len() != 1 {
                return Err(format!("layer {i}: layer_scalar has {} values", scalar.len()));
            }
            layers.push(LayerW {
                spec,
                input_norm: norm_buf(rt, st, &p("input_layernorm.weight"), h)?,
                post_attn_norm: norm_buf(rt, st, &p("post_attention_layernorm.weight"), h)?,
                pre_ffn_norm: norm_buf(rt, st, &p("pre_feedforward_layernorm.weight"), h)?,
                post_ffn_norm: norm_buf(rt, st, &p("post_feedforward_layernorm.weight"), h)?,
                w_qkv: pack_linear(
                    rt,
                    st,
                    &[
                        (p("self_attn.q_proj.weight"), q * d),
                        (p("self_attn.k_proj.weight"), kv * d),
                        (p("self_attn.v_proj.weight"), kv * d),
                    ],
                    h,
                )?,
                w_o: pack_linear(rt, st, &[(p("self_attn.o_proj.weight"), h)], q * d)?,
                q_norm: norm_buf(rt, st, &p("self_attn.q_norm.weight"), d)?,
                k_norm: norm_buf(rt, st, &p("self_attn.k_norm.weight"), d)?,
                gate_up: pack_linear(
                    rt,
                    st,
                    &[(p("mlp.gate_proj.weight"), inter), (p("mlp.up_proj.weight"), inter)],
                    h,
                )?,
                down: pack_linear(rt, st, &[(p("mlp.down_proj.weight"), h)], inter)?,
                ple_gate: pack_linear(rt, st, &[(p("ple_block.per_layer_input_gate.weight"), ple)], h)?,
                ple_out: pack_linear(rt, st, &[(p("ple_block.per_layer_projection.weight"), h)], ple)?,
                post_ple_norm: norm_buf(rt, st, &p("ple_block.post_per_layer_input_norm.weight"), h)?,
                layer_scalar: scalar[0],
            });
        }
        rt.synchronize()?;
        Ok(Self {
            embed,
            ple_w,
            ple_norm,
            final_norm,
            projection,
            layers,
        })
    }
}

// ---------------------------------------------------------------------------
// Activations: the library's `Acts`, by the same formulas
// ---------------------------------------------------------------------------

/// `(name, elements)` of every activation `EmbedGemma2Model::acts(nb, rows)`
/// allocates (src/embedgemma2.rs `acts`), by the same formulas.
fn acts_layout(cfg: &EmbedGemma2Config, nb: usize, rows: usize, ple_run_bytes: usize) -> Vec<(&'static str, usize)> {
    let r = rows.max(1);
    let nb = nb.max(1);
    let (h, ple, inter) = (cfg.hidden as usize, cfg.ple_dim as usize, cfg.intermediate as usize);
    let (mut qkv_w, mut q_w, mut kv_w) = (0usize, 0usize, 0usize);
    for s in &cfg.layers {
        let (q, kv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
        qkv_w = qkv_w.max((q + 2 * kv) * d);
        q_w = q_w.max(q * d);
        kv_w = kv_w.max(kv * d);
    }
    vec![
        ("embeds", r * h),
        ("resid", r * h),
        ("x", r * h),
        ("q", r * q_w),
        ("k", r * kv_w),
        ("v", r * kv_w),
        ("attn (+mlp mid)", r * q_w.max(inter)),
        ("y", r * h),
        // One buffer: the q|k|v GEMM's output, then the MLP's gate|up
        // (`Acts::qkv` is a clone of `Acts::gate_up`).
        ("gate_up (= qkv)", r * qkv_w.max(2 * inter)),
        (
            "ple run",
            (ple_run_bytes / 4).max(r * ple).min(r * ple * cfg.layers.len()),
        ),
        ("ple_gate", r * ple),
        ("ple_mid", r * ple),
        ("pooled", nb * h),
        // `ForwardIo`, per forward.
        ("io ids", r),
        ("io lens", nb),
        ("io segments", 2 * nb),
        ("io out", nb * cfg.embedding_dim as usize),
    ]
}

/// The same, for 2a7cb8b's `acts(nb, seq)` (`git show 2a7cb8b:src/embedgemma2.rs`
/// lines 1155-1210): separate gate/up/mid, per-width K, V and normed V, one
/// layer's per-layer input, ids/lens/segments and `out` inside it.
fn acts_layout_2a7cb8b(cfg: &EmbedGemma2Config, nb: usize, rows: usize) -> Vec<(&'static str, usize)> {
    let r = rows.max(1);
    let nb = nb.max(1);
    let (h, ple, inter) = (cfg.hidden as usize, cfg.ple_dim as usize, cfg.intermediate as usize);
    let (mut qkv_w, mut q_w) = (0usize, 0usize);
    let mut kv_widths: Vec<usize> = Vec::new();
    for s in &cfg.layers {
        let (q, kv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
        qkv_w = qkv_w.max((q + 2 * kv) * d);
        q_w = q_w.max(q * d);
        if !kv_widths.contains(&(kv * d)) {
            kv_widths.push(kv * d);
        }
    }
    let mut v = vec![
        ("ids", r),
        ("lens", nb),
        ("segments", 2 * nb),
        ("embeds", r * h),
        ("resid", r * h),
        ("x", r * h),
        ("qkv", r * qkv_w),
        ("q", r * q_w),
    ];
    for w in kv_widths {
        v.push(("k", r * w));
        v.push(("v", r * w));
        v.push(("v_normed", r * w));
    }
    v.extend([
        ("attn", r * q_w),
        ("y", r * h),
        ("gate", r * inter),
        ("up", r * inter),
        ("mid", r * inter),
        ("ple", r * ple),
        ("ple_gate", r * ple),
        ("ple_mid", r * ple),
        ("pooled", nb * h),
        ("out", nb * cfg.embedding_dim as usize),
    ]);
    v
}

struct ActsR {
    embeds: Tensor,
    resid: Tensor,
    x: Tensor,
    qkv: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn: Tensor,
    y: Tensor,
    gate_up: Tensor,
    ple: Tensor,
    ple_gate: Tensor,
    ple_mid: Tensor,
    pooled: Tensor,
}

impl ActsR {
    fn new(
        rt: &Arc<GpuRuntime>,
        cfg: &EmbedGemma2Config,
        nb: usize,
        rows: usize,
        ple_run_bytes: usize,
        zeroed: bool,
    ) -> Result<Self, String> {
        let lay = acts_layout(cfg, nb, rows, ple_run_bytes);
        let (h, ple, inter) = (cfg.hidden as usize, cfg.ple_dim as usize, cfg.intermediate as usize);
        let (mut qkv_w, mut q_w, mut kv_w) = (0usize, 0usize, 0usize);
        for s in &cfg.layers {
            let (q, kv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
            qkv_w = qkv_w.max((q + 2 * kv) * d);
            q_w = q_w.max(q * d);
            kv_w = kv_w.max(kv * d);
        }
        let r = rows.max(1);
        // Zeroed by default, unlike the library's: a skipped stage must leave finite data.
        let f = |shape: &[usize]| {
            if zeroed {
                rt.alloc_tensor_f32(shape)
            } else {
                rt.alloc_tensor_unzeroed(shape, DType::F32)
            }
        };
        let ple_n = lay.iter().find(|(n, _)| *n == "ple run").map_or(0, |p| p.1);
        Ok(Self {
            embeds: f(&[r, h])?,
            resid: f(&[r, h])?,
            x: f(&[r, h])?,
            qkv: f(&[r, qkv_w])?,
            q: f(&[r, q_w])?,
            k: f(&[r, kv_w])?,
            v: f(&[r, kv_w])?,
            attn: f(&[r, q_w.max(inter)])?,
            y: f(&[r, h])?,
            gate_up: f(&[r, 2 * inter])?,
            ple: f(&[ple_n])?,
            ple_gate: f(&[r, ple])?,
            ple_mid: f(&[r, ple])?,
            pooled: f(&[nb.max(1), h])?,
        })
    }
}

struct IoR {
    ids: GpuBuffer,
    lens: GpuBuffer,
    segments: GpuBuffer,
    out: Tensor,
}

impl IoR {
    fn new(rt: &Arc<GpuRuntime>, cfg: &EmbedGemma2Config, batch: &[Vec<u32>]) -> Result<Self, String> {
        let seq = batch.iter().map(Vec::len).max().unwrap_or(0);
        let rows = batch.len() * seq;
        let mut padded = vec![0u32; rows];
        for (b, ids) in batch.iter().enumerate() {
            padded[b * seq..b * seq + ids.len()].copy_from_slice(ids);
        }
        let lens: Vec<u32> = batch.iter().map(|i| i.len() as u32).collect();
        let segs: Vec<u32> = batch
            .iter()
            .enumerate()
            .flat_map(|(b, ids)| [(b * seq) as u32, (b * seq + ids.len()) as u32])
            .collect();
        Ok(Self {
            ids: rt.alloc_buffer_from_u32(&padded)?,
            lens: rt.alloc_buffer_from_u32(&lens)?,
            segments: rt.alloc_buffer_from_u32(&segs)?,
            out: rt.alloc_tensor_unzeroed(&[batch.len(), cfg.embedding_dim as usize], DType::F32)?,
        })
    }
}

// ---------------------------------------------------------------------------
// The replica forward
// ---------------------------------------------------------------------------

/// Per-stage device-memory accounting for `ledger` mode.
#[derive(Default)]
struct Ledger {
    /// Per class: (max fresh bytes a call allocated above its start, bytes still
    /// allocated after it above its start).
    per_class: Vec<(u64, i64)>,
    start: u64,
}

struct Cx<'a> {
    rt: &'a Arc<GpuRuntime>,
    cfg: &'a EmbedGemma2Config,
    w: &'a Weights,
    a: &'a ActsR,
    io: &'a IoR,
    nb: u32,
    seq: u32,
    ledger: Option<RefCell<Ledger>>,
}

impl Cx<'_> {
    fn before(&self) -> Result<(), String> {
        if let Some(l) = &self.ledger {
            self.rt.synchronize()?;
            self.rt.reset_peak_allocated_bytes();
            l.borrow_mut().start = self.rt.current_allocated_bytes();
        }
        Ok(())
    }

    fn after(&self, c: C) -> Result<(), String> {
        if let Some(l) = &self.ledger {
            self.rt.synchronize()?;
            let mut l = l.borrow_mut();
            if l.per_class.is_empty() {
                l.per_class = vec![(0, 0); N_CLASSES];
            }
            let start = l.start;
            let fresh = self.rt.peak_allocated_bytes().saturating_sub(start);
            let kept = self.rt.current_allocated_bytes() as i64 - start as i64;
            let e = &mut l.per_class[c as usize];
            e.0 = e.0.max(fresh);
            e.1 += kept;
        }
        Ok(())
    }
}

/// `body` when class `c` is on in `mask`, bracketed by the ledger hooks.
macro_rules! stage {
    ($cx:expr, $mask:expr, $c:expr, $body:block) => {
        if $mask & bit($c) != 0 {
            $cx.before()?;
            $body
            $cx.after($c)?;
        }
    };
}

/// `cx` again, without its ledger.
fn cx_copy<'a>(cx: &Cx<'a>) -> Cx<'a> {
    Cx {
        rt: cx.rt,
        cfg: cx.cfg,
        w: cx.w,
        a: cx.a,
        io: cx.io,
        nb: cx.nb,
        seq: cx.seq,
        ledger: None,
    }
}

fn view(t: &Tensor, rows: u32, cols: u32) -> Result<Tensor, String> {
    t.try_view(&[rows as usize, cols as usize], 0)
}

/// Queue one forward (src/embedgemma2.rs `forward`, op for op) over the
/// classes in `mask`; does not wait.
fn forward(cx: &Cx<'_>, mask: u32) -> Result<(), String> {
    let (rt, cfg, w, a, io) = (cx.rt, cx.cfg, cx.w, cx.a, cx.io);
    let (nb, seq) = (cx.nb, cx.seq);
    let rows = nb as usize * seq as usize;
    let (h, rows_u) = (cfg.hidden, rows as u32);
    let eps = cfg.rms_norm_eps;
    let at = |t: &Tensor, cols: u32| view(t, rows_u, cols);
    let (embeds, resid, x, y) = (at(&a.embeds, h)?, at(&a.resid, h)?, at(&a.x, h)?, at(&a.y, h)?);
    let inter = cfg.intermediate;
    let (gate_up, mid) = (at(&a.gate_up, 2 * inter)?, at(&a.attn, inter)?);
    let (ple_gate, ple_mid) = (at(&a.ple_gate, cfg.ple_dim)?, at(&a.ple_mid, cfg.ple_dim)?);
    let (n_layers, pd) = (w.layers.len(), cfg.ple_dim as usize);
    let run = (a.ple.numel() / (rows * pd)).clamp(1, n_layers);
    let mut ple_ld = 0u32;
    stage!(cx, mask, C::Gather, {
        qwen35::embed_rows_scaled(
            rt,
            &io.ids,
            rows_u,
            LmHead {
                weight: &w.embed.buffer,
                dtype: DType::BF16,
                vocab: cfg.vocab,
            },
            h,
            (h as f32).sqrt(),
            &embeds.buffer,
        )?;
        gpu_copy(&embeds, &resid)?;
    });
    let ple_scale = (h as f32).powf(-0.5);
    for (i, layer) in w.layers.iter().enumerate() {
        let s = layer.spec;
        let (c_qkv, c_attn, c_o) = if s.sliding {
            (C::QkvSliding, C::AttnSliding, C::OSliding)
        } else {
            (C::QkvGlobal, C::AttnGlobal, C::OGlobal)
        };
        stage!(cx, mask, C::RmsNorm, {
            rms_norm_f32(rt, &resid.buffer, &layer.input_norm, &x.buffer, rows_u, h, eps)?;
        });
        let qkv_width = (cfg.q_heads + 2 * s.kv_heads) * s.head_dim;
        let qkv = at(&a.qkv, qkv_width)?;
        stage!(cx, mask, c_qkv, {
            gemm(&x, &layer.w_qkv, &qkv, BACKEND)?;
        });
        stage!(cx, mask, C::QkNormRope, {
            qwen35::attn_qk_norm_rope_columns(
                rt,
                &AttnShape {
                    batch: nb,
                    seq,
                    q_heads: cfg.q_heads,
                    kv_heads: s.kv_heads,
                    head_dim: s.head_dim,
                    rotary_dim: s.head_dim,
                },
                Cols::dense(&qkv.buffer, qkv_width),
                QkvColumns {
                    q_head_stride: s.head_dim,
                    k_col: cfg.q_heads * s.head_dim,
                    v_col: (cfg.q_heads + s.kv_heads) * s.head_dim,
                    v_norm: true,
                },
                0.0,
                &layer.q_norm,
                &layer.k_norm,
                &AttnTargets {
                    q_out: &a.q.buffer,
                    k_cache: &a.k.buffer,
                    v_cache: &a.v.buffer,
                },
                DType::F32,
                0,
                s.rope_theta,
                eps,
            )?;
        });
        stage!(cx, mask, c_attn, {
            tessl::embedgemma2::encoder_attn(
                rt,
                &a.q.buffer,
                &a.k.buffer,
                &a.v.buffer,
                &a.attn.buffer,
                &io.lens,
                EncoderAttnDims {
                    batch: nb,
                    seq,
                    heads: cfg.q_heads,
                    heads_kv: s.kv_heads,
                    head_dim: s.head_dim,
                    window: if s.sliding { cfg.sliding_window } else { 0 },
                    scale: 1.0,
                },
            )?;
        });
        let attn = at(&a.attn, cfg.q_heads * s.head_dim)?;
        stage!(cx, mask, c_o, {
            gemm(&attn, &layer.w_o, &y, BACKEND)?;
        });
        stage!(cx, mask, C::NormResidAttn, {
            rms_norm_residual_add_f32(rt, &y.buffer, &layer.post_attn_norm, &resid.buffer, rows_u, h, eps, 1.0)?;
        });
        stage!(cx, mask, C::RmsNorm, {
            rms_norm_f32(rt, &resid.buffer, &layer.pre_ffn_norm, &x.buffer, rows_u, h, eps)?;
        });
        stage!(cx, mask, C::GateUp, {
            gemm(&x, &layer.gate_up, &gate_up, BACKEND)?;
        });
        let window = |off: u32| Cols {
            buf: &gate_up.buffer,
            ld: 2 * inter,
            off,
        };
        stage!(cx, mask, C::GeluMlp, {
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
        });
        stage!(cx, mask, C::Down, {
            gemm(&mid, &layer.down, &y, BACKEND)?;
        });
        stage!(cx, mask, C::NormResidFfn, {
            rms_norm_residual_add_f32(rt, &y.buffer, &layer.post_ffn_norm, &resid.buffer, rows_u, h, eps, 1.0)?;
        });
        if i % run == 0 {
            let k = run.min(n_layers - i);
            ple_ld = (k * pd) as u32;
            let wv = w.ple_w.try_view(&[k * pd, h as usize], i * pd * h as usize)?;
            let out = at(&a.ple, ple_ld)?;
            stage!(cx, mask, C::PleProj, {
                gemm_nt_f32(&embeds, &wv, &out, BACKEND)?;
            });
            stage!(cx, mask, C::PleScaleNorm, {
                scale_f32_inplace(rt, &out.buffer, ple_scale, rows_u * ple_ld)?;
                rms_norm_f32(
                    rt,
                    &out.buffer,
                    &w.ple_norm,
                    &out.buffer,
                    rows_u * k as u32,
                    cfg.ple_dim,
                    eps,
                )?;
            });
        }
        stage!(cx, mask, C::PleGate, {
            gemm(&resid, &layer.ple_gate, &ple_gate, BACKEND)?;
        });
        stage!(cx, mask, C::PleGelu, {
            qwen35::gated_act(
                rt,
                GatedAct::GeluTanh,
                Cols::dense(&ple_gate.buffer, cfg.ple_dim),
                Cols {
                    buf: &a.ple.buffer,
                    ld: ple_ld,
                    off: ((i % run) * pd) as u32,
                },
                OutCols {
                    cols: Cols::dense(&ple_mid.buffer, cfg.ple_dim),
                    dtype: DType::F32,
                },
                rows_u,
                cfg.ple_dim,
            )?;
        });
        stage!(cx, mask, C::PleOut, {
            gemm(&ple_mid, &layer.ple_out, &y, BACKEND)?;
        });
        stage!(cx, mask, C::NormResidPle, {
            rms_norm_residual_add_f32(
                rt,
                &y.buffer,
                &layer.post_ple_norm,
                &resid.buffer,
                rows_u,
                h,
                eps,
                layer.layer_scalar,
            )?;
        });
    }
    stage!(cx, mask, C::Tail, {
        let (pooled, out) = (view(&a.pooled, nb, h)?, view(&io.out, nb, cfg.embedding_dim)?);
        rms_norm_f32(rt, &resid.buffer, &w.final_norm, &x.buffer, rows_u, h, eps)?;
        segment_mean_rows(rt, &x.buffer, &io.segments, &pooled.buffer, nb, rows_u, h)?;
        gemm(&pooled, &w.projection, &out, BACKEND)?;
        l2_normalize_rows(rt, &out.buffer, nb, cfg.embedding_dim, cfg.embedding_dim)?;
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Work estimates per class and forward (for achieved GB/s and TFLOP/s)
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

/// Query-key pairs one sequence of `seq` tokens visits per head: all of them
/// (`window == 0`), or `|i - j| <= window`.
fn pairs(seq: usize, window: usize) -> f64 {
    if window == 0 || window >= seq {
        return (seq * seq) as f64;
    }
    (0..seq)
        .map(|i| (seq - 1).min(i + window) - i.saturating_sub(window) + 1)
        .sum::<usize>() as f64
}

fn class_costs(cfg: &EmbedGemma2Config, nb: usize, seq: usize, run: usize) -> [Cost; N_CLASSES] {
    let rows = nb * seq;
    let (h, inter, pd) = (cfg.hidden as usize, cfg.intermediate as usize, cfg.ple_dim as usize);
    let l = cfg.layers.len();
    let q = cfg.q_heads as usize;
    let mut c = [Cost::default(); N_CLASSES];
    let el = |n: usize| 4.0 * n as f64;
    c[C::Gather as usize] = Cost {
        flops: 0.0,
        bytes: (rows * h) as f64 * (2.0 + 4.0 + 8.0),
    };
    c[C::RmsNorm as usize] = Cost {
        flops: 0.0,
        bytes: 2.0 * l as f64 * el(2 * rows * h),
    };
    for s in &cfg.layers {
        let (d, kv) = (s.head_dim as usize, s.kv_heads as usize);
        let (ci_qkv, ci_attn, ci_o) = if s.sliding {
            (C::QkvSliding, C::AttnSliding, C::OSliding)
        } else {
            (C::QkvGlobal, C::AttnGlobal, C::OGlobal)
        };
        let add = |dst: &mut Cost, x: Cost| {
            dst.flops += x.flops;
            dst.bytes += x.bytes;
        };
        add(&mut c[ci_qkv as usize], gemm_cost(rows, h, (q + 2 * kv) * d, 1));
        add(&mut c[ci_o as usize], gemm_cost(rows, q * d, h, 1));
        add(
            &mut c[C::QkNormRope as usize],
            Cost {
                flops: 0.0,
                bytes: el(rows * (q + 2 * kv) * d) + el(rows * (q + 2 * kv) * d),
            },
        );
        let win = if s.sliding { cfg.sliding_window as usize } else { 0 };
        add(
            &mut c[ci_attn as usize],
            Cost {
                flops: 4.0 * (nb * q * d) as f64 * pairs(seq, win),
                bytes: el(2 * rows * q * d) + el(2 * rows * kv * d),
            },
        );
    }
    c[C::NormResidAttn as usize] = Cost {
        flops: 0.0,
        bytes: l as f64 * el(3 * rows * h),
    };
    c[C::GateUp as usize] = gemm_cost(rows, h, 2 * inter, l);
    c[C::GeluMlp as usize] = Cost {
        flops: 0.0,
        bytes: l as f64 * el(rows * 3 * inter),
    };
    c[C::Down as usize] = gemm_cost(rows, inter, h, l);
    c[C::NormResidFfn as usize] = c[C::NormResidAttn as usize];
    let runs = l.div_ceil(run);
    c[C::PleProj as usize] = Cost {
        flops: 2.0 * (rows * h * l * pd) as f64,
        bytes: el(l * pd * h) + runs as f64 * el(rows * h) + el(rows * l * pd),
    };
    c[C::PleScaleNorm as usize] = Cost {
        flops: 0.0,
        bytes: el(4 * rows * l * pd),
    };
    c[C::PleGate as usize] = gemm_cost(rows, h, pd, l);
    c[C::PleGelu as usize] = Cost {
        flops: 0.0,
        bytes: l as f64 * el(rows * 3 * pd),
    };
    c[C::PleOut as usize] = gemm_cost(rows, pd, h, l);
    c[C::NormResidPle as usize] = c[C::NormResidAttn as usize];
    c[C::Tail as usize] = Cost {
        flops: 2.0 * (nb * h * cfg.embedding_dim as usize) as f64,
        bytes: el(2 * rows * h) + el(rows * h) + el(h * cfg.embedding_dim as usize),
    };
    c
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

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

fn parse_workload(w: &str) -> Result<(usize, usize), String> {
    let (b, t) = w
        .trim()
        .split_once('x')
        .ok_or_else(|| format!("workload {w:?} is not BxT"))?;
    let num = |s: &str| s.parse::<usize>().map_err(|e| format!("workload {w:?}: {s:?}: {e}"));
    let (b, t) = (num(b)?, num(t)?);
    if b == 0 || t < 2 {
        return Err(format!("workload {w:?} needs B >= 1 and T >= 2"));
    }
    Ok((b, t))
}

fn snapshot() -> Result<PathBuf, String> {
    std::env::var("EMBEDGEMMA2_SNAPSHOT")
        .map(PathBuf::from)
        .map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory".to_string())
}

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.6}")
    } else {
        "null".to_string()
    }
}

fn jarr(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|&x| jnum(x)).collect::<Vec<_>>().join(","))
}

fn jbools(v: &[bool]) -> String {
    format!("[{}]", v.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","))
}

fn jstr(s: &str) -> String {
    format!("{s:?}")
}

fn min_of(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn median_of(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

/// The device's GPU utilization as `ioreg` reports it, for the machine-state
/// record (other sessions share this GPU).
fn gpu_utilization() -> String {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.split("\"Device Utilization %\"=")
                .nth(1)
                .and_then(|t| t.split(|c: char| !c.is_ascii_digit()).next())
                .unwrap_or("unknown")
                .to_string()
        }
        Err(e) => format!("ioreg failed: {e}"),
    }
}

fn write_json(path: &str, body: &str) -> Result<(), String> {
    common::atomic_write(std::path::Path::new(path), body.as_bytes())
}

// ---------------------------------------------------------------------------
// Mode: stages
// ---------------------------------------------------------------------------

struct Variant {
    name: String,
    /// Class index for `only:`/`skip:` variants.
    class: Option<usize>,
    kind: &'static str,
    mask: u32,
    /// The samples the statistics use: those taken with no other tessl GPU
    /// process running (every kept sample when none was clean).
    samples_ms: Vec<f64>,
    enc_ms: Vec<f64>,
    /// Every kept sample, with whether another tessl GPU process was seen
    /// before or after it (see [`others_running`]).
    raw_ms: Vec<f64>,
    raw_enc_ms: Vec<f64>,
    contended: Vec<bool>,
    /// Samples thrown away and retaken because another process appeared.
    resampled: usize,
    dispatches: usize,
}

/// Other tessl processes that may be using the GPU: test and bench binaries
/// under a `tessl*` checkout's `target/`, other than this one. Other sessions
/// share this machine; a sample taken while one runs is flagged, not trusted.
/// A GPU user this does not see (a display, a browser) stays in the baseline,
/// which the `ioreg` utilization reading records.
fn others_running() -> Vec<String> {
    /// `PROC_ALL_PIDS` of `<sys/proc_info.h>`; the `libc` crate does not export it.
    const PROC_ALL_PIDS: u32 = 1;
    let me = std::process::id() as libc::pid_t;
    // SAFETY: with a null buffer `proc_listpids` only returns the bytes needed.
    let need = unsafe { libc::proc_listpids(PROC_ALL_PIDS, 0, std::ptr::null_mut(), 0) };
    if need <= 0 {
        return vec!["proc_listpids failed".to_string()];
    }
    let mut pids = vec![0 as libc::pid_t; need as usize / std::mem::size_of::<libc::pid_t>() + 64];
    let bytes = (pids.len() * std::mem::size_of::<libc::pid_t>()) as libc::c_int;
    // SAFETY: `pids` is a live buffer of `bytes` bytes.
    let got = unsafe { libc::proc_listpids(PROC_ALL_PIDS, 0, pids.as_mut_ptr().cast(), bytes) };
    if got <= 0 {
        return vec!["proc_listpids failed".to_string()];
    }
    let n = got as usize / std::mem::size_of::<libc::pid_t>();
    let mut found = Vec::new();
    for &pid in &pids[..n.min(pids.len())] {
        if pid <= 0 || pid == me {
            continue;
        }
        let mut buf = [0u8; 4096];
        // SAFETY: `buf` is a live buffer of the length passed.
        let len = unsafe { libc::proc_pidpath(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
        if len <= 0 {
            continue;
        }
        let path = String::from_utf8_lossy(&buf[..len as usize]).into_owned();
        let in_target =
            path.contains("/research/tessl") && (path.contains("/target/release/") || path.contains("/target/debug/"));
        let file = path.rsplit('/').next().unwrap_or("");
        let gpu_bin = path.contains("/deps/")
            || file.starts_with("bench_")
            || file.starts_with("probe_")
            || file.starts_with("profile_");
        if in_target && gpu_bin {
            found.push(file.to_string());
        }
    }
    found
}

/// Milliseconds this process has spent waiting for a quiet machine.
static WAITED_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Wait for [`others_running`] to be empty, for at most `PROFILE_WAIT_S`
/// seconds (default 1800) in total over the whole run; false if it was not
/// empty when the budget ran out.
fn wait_quiet() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let budget_ms = 1000
        * std::env::var("PROFILE_WAIT_S")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1800);
    while !others_running().is_empty() {
        if WAITED_MS.load(Relaxed) >= budget_ms {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(1000));
        WAITED_MS.fetch_add(1000, Relaxed);
    }
    true
}

/// `run` once the machine is quiet, retaken (up to 3 times) when another tessl
/// GPU process appears during it: (result, contended, retakes).
fn gated<T>(mut run: impl FnMut() -> Result<T, String>) -> Result<(T, bool, usize), String> {
    let mut retakes = 0;
    loop {
        let quiet = wait_quiet();
        let r = run()?;
        let seen_after = !others_running().is_empty();
        if !seen_after && quiet {
            return Ok((r, false, retakes));
        }
        if retakes >= 3 || !quiet {
            return Ok((r, true, retakes));
        }
        retakes += 1;
    }
}

/// The samples of `v` taken while the machine was quiet, or all of them when
/// none was.
fn clean_or_all(v: &[f64], contended: &[bool]) -> Vec<f64> {
    let clean: Vec<f64> = v.iter().zip(contended).filter(|(_, &c)| !c).map(|(&x, _)| x).collect();
    if clean.is_empty() {
        v.to_vec()
    } else {
        clean
    }
}

/// `reps` forwards queued, one wait: (total ms per forward, encode ms per forward).
fn timed_reps(rt: &Arc<GpuRuntime>, cx: &Cx<'_>, mask: u32, reps: usize) -> Result<(f64, f64), String> {
    let t0 = Instant::now();
    for _ in 0..reps {
        forward(cx, mask)?;
    }
    let enc = t0.elapsed();
    rt.synchronize()?;
    let tot = t0.elapsed();
    Ok((
        tot.as_secs_f64() * 1e3 / reps as f64,
        enc.as_secs_f64() * 1e3 / reps as f64,
    ))
}

fn mode_stages() -> Result<(), String> {
    let snap = snapshot()?;
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    let workloads = spec.split(',').map(parse_workload).collect::<Result<Vec<_>, _>>()?;
    let rounds = env_usize("PROFILE_ROUNDS", 5, 1)?;
    // `PROFILE_SKIP=0` leaves out the `skip:` variants, half the run time.
    let with_skip = std::env::var("PROFILE_SKIP").as_deref() != Ok("0");
    let out_path = std::env::var("PROFILE_JSON").unwrap_or_else(|_| "target/profile_stages.json".into());

    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(true)?;
    let model = EmbedGemma2Model::load(&rt, &st, PREFIX, cfg.clone())?;
    let weights = Weights::load(&rt, &st, &cfg)?;
    let util0 = gpu_utilization();
    eprintln!("gpu utilization at start: {util0}%");

    let mut json_wl = Vec::new();
    for &(b, t) in &workloads {
        let name = format!("{b}x{t}");
        let seqs: Vec<Vec<u32>> = (0..b).map(|s| ids(s, t)).collect();
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        let rows = b * t;
        let acts = ActsR::new(&rt, &cfg, b, rows, tessl::embedgemma2::PLE_RUN_BYTES, true)?;
        let io = IoR::new(&rt, &cfg, &seqs)?;
        let cx = Cx {
            rt: &rt,
            cfg: &cfg,
            w: &weights,
            a: &acts,
            io: &io,
            nb: b as u32,
            seq: t as u32,
            ledger: None,
        };
        let run = (acts.ple.numel() / (rows * cfg.ple_dim as usize)).clamp(1, cfg.layers.len());
        let costs = class_costs(&cfg, b, t, run);

        // Validation: the replica reproduces the library's embeddings.
        let lib = model.encode(&batch, None, false)?;
        timed_reps(&rt, &cx, ALL, 1)?;
        let dim = cfg.embedding_dim as usize;
        let mine: Vec<f32> = io.out.buffer.try_contents_f32()?[..b * dim].to_vec();
        let bit_equal = mine
            .iter()
            .zip(&lib.embeddings)
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count();
        let max_abs = mine
            .iter()
            .zip(&lib.embeddings)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{name}: replica vs encode: {bit_equal}/{} values bit-equal, max abs diff {max_abs:e}",
            mine.len()
        );
        if max_abs > 1e-5 {
            return Err(format!(
                "{name}: the replica forward disagrees with encode by {max_abs}"
            ));
        }

        // Dispatches per class and per forward.
        let mut variants: Vec<Variant> = Vec::new();
        let mut mk = |name: String, class: Option<usize>, kind: &'static str, mask: u32| -> Result<(), String> {
            rt.synchronize()?;
            rt.take_dispatch_count();
            forward(&cx, mask)?;
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
        mk(
            "full, fresh unzeroed activations per call (as encode)".into(),
            None,
            "fresh_unzeroed",
            ALL,
        )?;
        mk(
            "full, fresh zeroed activations per call".into(),
            None,
            "fresh_zeroed",
            ALL,
        )?;
        for (c, nm) in NAMES.iter().enumerate() {
            mk(format!("only:{nm}"), Some(c), "only", 1 << c)?;
            if with_skip {
                mk(format!("skip:{nm}"), Some(c), "skip", ALL & !(1 << c))?;
            }
        }
        let dispatch_full = variants[0].dispatches;

        // Forwards queued per wait: enough that the wait is a small part of a
        // sample, few enough that the constant arena never forces a host wait.
        let probe = {
            timed_reps(&rt, &cx, ALL, 1)?;
            timed_reps(&rt, &cx, ALL, 1)?.0
        };
        let reps = ((80.0 / probe).ceil() as usize)
            .clamp(1, 16)
            .min((8000 / dispatch_full.max(1)).max(1));
        eprintln!("{name}: {dispatch_full} dispatches per forward, {reps} forward(s) per wait, probe {probe:.2} ms");

        let n_var = variants.len() + 1; // + encode
        for v in &variants {
            timed_reps(&rt, &cx, v.mask, reps.min(2))?; // warm-up
        }
        model.encode(&batch, None, false)?;
        let mut encode_raw = Vec::with_capacity(rounds);
        let mut encode_contended = Vec::with_capacity(rounds);
        for round in 0..rounds {
            for i in round_order(round, n_var) {
                if i == variants.len() {
                    let (ms, contended, _) = gated(|| {
                        let t0 = Instant::now();
                        model.encode(&batch, None, false)?;
                        Ok(t0.elapsed().as_secs_f64() * 1e3)
                    })?;
                    encode_raw.push(ms);
                    encode_contended.push(contended);
                } else {
                    let (mask, kind) = (variants[i].mask, variants[i].kind);
                    let ((tot, enc), contended, retakes) = gated(|| match kind {
                        "fresh_unzeroed" | "fresh_zeroed" => {
                            // One forward on a new activation set, as `encode` makes.
                            let t0 = Instant::now();
                            let a2 = ActsR::new(
                                &rt,
                                &cfg,
                                b,
                                rows,
                                tessl::embedgemma2::PLE_RUN_BYTES,
                                kind == "fresh_zeroed",
                            )?;
                            let cx2 = Cx { a: &a2, ..cx_copy(&cx) };
                            forward(&cx2, ALL)?;
                            let enc = t0.elapsed();
                            rt.synchronize()?;
                            let tot = t0.elapsed();
                            drop(cx2);
                            drop(a2);
                            Ok((tot.as_secs_f64() * 1e3, enc.as_secs_f64() * 1e3))
                        }
                        _ => timed_reps(&rt, &cx, mask, reps),
                    })?;
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
        println!("== workload {name} ({rows} rows, {dispatch_full} dispatches/forward, {reps} forwards per wait, {rounds} rounds)");
        println!(
            "   full replica {full:.3} ms (min) / {:.3} (median); encode() {enc_min:.3} ms (min); empty wait {none:.3} ms; host encode time of full {:.3} ms",
            median_of(&variants[0].samples_ms),
            min_of(&variants[0].enc_ms)
        );
        for v in variants.iter().filter(|v| v.kind.starts_with("fresh")) {
            println!("   {}: {:.3} ms (min)", v.name, min_of(&v.samples_ms));
        }
        println!(
            "   {:<34} {:>5} {:>9} {:>6} {:>10} {:>6} {:>9} {:>8} {:>8}",
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
                "   {:<34} {:>5} {:>9.3} {:>5.1}% {:>10} {:>6} {:>9.2} {:>8.0} {:>8}",
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
                    "{{\"name\":{},\"kind\":{},\"class\":{},\"mask\":{},\"dispatches\":{},\"ms_per_forward\":{},\"host_encode_ms_per_forward\":{},\"ms_min\":{},\"raw_ms_per_forward\":{},\"contended\":{},\"resampled\":{}}}",
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
            "{{\"workload\":{},\"rows\":{rows},\"batch\":{b},\"seq\":{t},\"reps_per_wait\":{reps},\"rounds\":{rounds},\"dispatches_full\":{dispatch_full},\"ple_run_layers\":{run},\"replica_vs_encode\":{{\"values\":{},\"bit_equal\":{bit_equal},\"max_abs_diff\":{max_abs:e}}},\"encode_ms\":{},\"encode_raw_ms\":{},\"encode_contended\":{},\"samples_contended\":{n_contended},\"samples_total\":{n_samples},\"samples_resampled\":{n_resampled},\"gpu_util_start_pct\":{},\"gpu_util_end_pct\":{},\"costs\":[{}],\"variants\":[{}]}}",
            jstr(&name),
            mine.len(),
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
    write_json(&out_path, &body)?;
    eprintln!("wrote {out_path}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Mode: gemm (what-if bf16, machine ceilings)
// ---------------------------------------------------------------------------

fn bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn rand_bf16(rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_bf16(shape)?;
    let n: usize = shape.iter().product();
    let data = common::fill(n, seed);
    let mut dst = t.buffer.try_contents_u16()?;
    for (d, &s) in dst.iter_mut().zip(&data) {
        *d = bf16_bits(s);
    }
    drop(dst);
    Ok(t)
}

fn rand_f32(rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.try_write_f32(&common::fill(shape.iter().product(), seed))?;
    Ok(t)
}

/// Min-of-`rounds` ms per call of `f` run `reps` times per wait.
fn bench_calls(
    rt: &Arc<GpuRuntime>,
    rounds: usize,
    reps: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<Vec<f64>, String> {
    for _ in 0..2 {
        f()?;
    }
    rt.synchronize()?;
    let mut out = Vec::new();
    let mut contended = Vec::new();
    for _ in 0..rounds {
        let (ms, c, _) = gated(|| {
            let t0 = Instant::now();
            for _ in 0..reps {
                f()?;
            }
            rt.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
        })?;
        out.push(ms);
        contended.push(c);
    }
    if contended.iter().any(|&c| c) {
        eprintln!(
            "note: {} of {rounds} samples had another tessl GPU process running; using the clean ones",
            contended.iter().filter(|&&c| c).count()
        );
    }
    Ok(clean_or_all(&out, &contended))
}

fn mode_gemm() -> Result<(), String> {
    let rounds = env_usize("PROFILE_ROUNDS", 7, 1)?;
    let out_path = std::env::var("PROFILE_JSON").unwrap_or_else(|_| "target/profile_gemm.json".into());
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(true)?;
    let util = gpu_utilization();
    let mut json = Vec::new();

    // Ceilings: copy bandwidth, and large square GEMM rates.
    let n = 1usize << 26; // 256 MiB of f32
    let (src, dst) = (rand_f32(&rt, &[n], 1)?, rand_f32(&rt, &[n], 2)?);
    let copy = bench_calls(&rt, rounds, 8, || gpu_copy(&src, &dst))?;
    let copy_gbs = 2.0 * (n * 4) as f64 / (min_of(&copy) * 1e-3) / 1e9;
    let buf = rt.alloc_buffer(n * 4)?;
    let scale = bench_calls(&rt, rounds, 8, || scale_f32_inplace(&rt, &buf, 1.0, n as u32))?;
    let scale_gbs = 2.0 * (n * 4) as f64 / (min_of(&scale) * 1e-3) / 1e9;
    let s = 4096usize;
    let (a, b, c) = (
        rand_f32(&rt, &[s, s], 3)?,
        rand_f32(&rt, &[s, s], 4)?,
        rt.alloc_tensor_f32(&[s, s])?,
    );
    let f32_sq = bench_calls(&rt, rounds, 4, || gemm(&a, &b, &c, BACKEND))?;
    let f32_tf = 2.0 * (s * s * s) as f64 / (min_of(&f32_sq) * 1e-3) / 1e12;
    let (ab, bb) = (rand_bf16(&rt, &[s, s], 5)?, rand_bf16(&rt, &[s, s], 6)?);
    let bf_sq = bench_calls(&rt, rounds, 4, || gemm_bf16(&ab, &bb, &c))?;
    let bf_tf = 2.0 * (s * s * s) as f64 / (min_of(&bf_sq) * 1e-3) / 1e12;
    println!("ceilings (measured on this machine, GPU util {util}%): gpu_copy {copy_gbs:.0} GB/s, in-place scale {scale_gbs:.0} GB/s, f32 GEMM 4096^3 {f32_tf:.2} TFLOP/s, bf16-operand GEMM 4096^3 {bf_tf:.2} TFLOP/s");
    json.push(format!(
        "\"ceilings\":{{\"copy_gbs\":{},\"copy_ms\":{},\"scale_gbs\":{},\"scale_ms\":{},\"f32_gemm_4096_tflops\":{},\"f32_ms\":{},\"bf16_gemm_4096_tflops\":{},\"bf16_ms\":{}}}",
        jnum(copy_gbs),
        jarr(&copy),
        jnum(scale_gbs),
        jarr(&scale),
        jnum(f32_tf),
        jarr(&f32_sq),
        jnum(bf_tf),
        jarr(&bf_sq)
    ));
    drop((src, dst, buf, a, b, c, ab, bb));

    // The forward's GEMM shapes, one distinct weight per layer cycled.
    let shapes: [(&str, usize, usize); 8] = [
        ("qkv sliding", 512, 2048),
        ("qkv global", 512, 3072),
        ("o sliding", 1024, 512),
        ("o global", 2048, 512),
        ("gate|up", 512, 4096),
        ("down", 2048, 512),
        ("ple_gate", 512, 512),
        ("ple_out", 512, 512),
    ];
    let layers = 24usize;
    let mut rows_json = Vec::new();
    println!(
        "{:<12} {:>6} {:>5} {:>5} | {:>9} {:>7} {:>7} | {:>9} {:>7} {:>7} | {:>6}",
        "gemm", "rows", "k", "n", "f32 us", "TFLOP/s", "GB/s", "bf16 us", "TFLOP/s", "GB/s", "f32/bf16"
    );
    for &rows in &[16usize, 2048, 8192] {
        for (si, &(name, k, nn)) in shapes.iter().enumerate() {
            let a32 = rand_f32(&rt, &[rows, k], 10 + si as u64)?;
            let abf = rand_bf16(&rt, &[rows, k], 20 + si as u64)?;
            let c = rt.alloc_tensor_f32(&[rows, nn])?;
            let w32: Vec<Tensor> = (0..layers)
                .map(|l| rand_f32(&rt, &[k, nn], 100 + l as u64))
                .collect::<Result<_, _>>()?;
            let wbf: Vec<Tensor> = (0..layers)
                .map(|l| rand_bf16(&rt, &[k, nn], 200 + l as u64))
                .collect::<Result<_, _>>()?;
            let mut i = 0usize;
            let reps = 8 * layers;
            let t32 = bench_calls(&rt, rounds, reps, || {
                i += 1;
                gemm(&a32, &w32[i % layers], &c, BACKEND)
            })?;
            let tbf = bench_calls(&rt, rounds, reps, || {
                i += 1;
                gemm_bf16(&abf, &wbf[i % layers], &c)
            })?;
            let flops = 2.0 * (rows * k * nn) as f64;
            let bytes32 = 4.0 * (k * nn + rows * k + rows * nn) as f64;
            let bytesbf = (2 * (k * nn + rows * k) + 4 * rows * nn) as f64;
            let (m32, mbf) = (min_of(&t32), min_of(&tbf));
            println!(
                "{name:<12} {rows:>6} {k:>5} {nn:>5} | {:>9.2} {:>7.2} {:>7.0} | {:>9.2} {:>7.2} {:>7.0} | {:>6.2}",
                m32 * 1e3,
                flops / (m32 * 1e-3) / 1e12,
                bytes32 / (m32 * 1e-3) / 1e9,
                mbf * 1e3,
                flops / (mbf * 1e-3) / 1e12,
                bytesbf / (mbf * 1e-3) / 1e9,
                m32 / mbf
            );
            rows_json.push(format!(
                "{{\"gemm\":{},\"rows\":{rows},\"k\":{k},\"n\":{nn},\"flops\":{flops:.0},\"bytes_f32\":{bytes32:.0},\"bytes_bf16_operands\":{bytesbf:.0},\"f32_ms\":{},\"bf16_ms\":{}}}",
                jstr(name),
                jarr(&t32),
                jarr(&tbf)
            ));
        }
    }
    json.push(format!("\"gemm_shapes\":[{}]", rows_json.join(",")));

    // Per-dispatch fixed cost: a 64-element in-place scale.
    let tiny = rt.alloc_buffer(64 * 4)?;
    let async_tiny = bench_calls(&rt, rounds, 2000, || scale_f32_inplace(&rt, &tiny, 1.0, 64))?;
    rt.set_async_encode(false)?;
    let sync_tiny = bench_calls(&rt, rounds.min(5), 100, || scale_f32_inplace(&rt, &tiny, 1.0, 64))?;
    rt.set_async_encode(true)?;
    println!(
        "dispatch cost, 64-element scale: queued {:.2} us/dispatch (min of {rounds} x 2000 per wait), commit+wait each {:.1} us (min of {} x 100)",
        min_of(&async_tiny) * 1e3,
        min_of(&sync_tiny) * 1e3,
        rounds.min(5)
    );
    json.push(format!(
        "\"dispatch_cost\":{{\"queued_ms_per_dispatch\":{},\"sync_ms_per_dispatch\":{}}}",
        jarr(&async_tiny),
        jarr(&sync_tiny)
    ));
    let body = format!(
        "{{\"mode\":\"gemm\",\"gpu_util_pct\":{},{}}}",
        jstr(&util),
        json.join(",")
    );
    write_json(&out_path, &body)?;
    eprintln!("wrote {out_path}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Mode: memory
// ---------------------------------------------------------------------------

fn mode_memory() -> Result<(), String> {
    let snap = snapshot()?;
    let wl = std::env::var("PROFILE_WORKLOAD").unwrap_or_else(|_| "8x1024".into());
    let (b, t) = parse_workload(&wl)?;
    let steps = std::env::var("PROFILE_STEPS").unwrap_or_else(|_| "E,E,S,E".into());
    let async_encode = std::env::var("BENCH_ASYNC").as_deref() == Ok("1");
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(async_encode)?;
    let operands = match std::env::var("PROFILE_OPERANDS").as_deref() {
        Ok("f32") | Err(_) => GemmOperands::ExactF32,
        Ok("bf16") => GemmOperands::Bf16,
        Ok(other) => return Err(format!("PROFILE_OPERANDS={other:?}: expected f32 or bf16")),
    };
    let mut model = EmbedGemma2Model::load_with(&rt, &st, PREFIX, cfg.clone(), operands)?;
    let ple_bytes = match std::env::var("PROFILE_PLE_BYTES") {
        Ok(v) => {
            let n: usize = v.parse().map_err(|e| format!("PROFILE_PLE_BYTES={v:?}: {e}"))?;
            model.set_ple_run_bytes(n)?;
            n
        }
        Err(_) => tessl::embedgemma2::PLE_RUN_BYTES,
    };
    let loaded = rt.current_allocated_bytes();
    let seqs: Vec<Vec<u32>> = (0..b).map(|s| ids(s, t)).collect();
    let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let rows = b * t;
    let mut held: Vec<Tensor> = Vec::new();
    let mut recs = Vec::new();
    let mut cum_peak = 0u64;
    // `bench_embedgemma2` runs its workloads in one process and takes the peak
    // from each workload's start, which includes what earlier workloads left in
    // the pool's freelist; `PROFILE_PRELUDE` (workloads, each encoded twice)
    // reproduces that.
    if let Ok(pre) = std::env::var("PROFILE_PRELUDE") {
        for w in pre.split(',').filter(|w| !w.trim().is_empty()) {
            let (pb, pt) = parse_workload(w)?;
            let s: Vec<Vec<u32>> = (0..pb).map(|s| ids(s, pt)).collect();
            let bt: Vec<&[u32]> = s.iter().map(Vec::as_slice).collect();
            for _ in 0..2 {
                model.encode(&bt, None, false)?;
            }
        }
    }
    let prelude_cached = rt.current_allocated_bytes();
    rt.reset_peak_allocated_bytes();
    let mut forwards = 0;
    for (n, step) in steps.split(',').enumerate() {
        let step = step.trim();
        let before = rt.current_allocated_bytes();
        match step {
            "E" => forwards = model.encode(&batch, None, false)?.forwards,
            "S" => rt.synchronize()?,
            "A" | "O" => {
                let lay = if step == "A" {
                    acts_layout(&cfg, b, rows, ple_bytes)
                } else {
                    acts_layout_2a7cb8b(&cfg, b, rows)
                };
                for (_, el) in lay {
                    held.push(rt.alloc_tensor_unzeroed(&[el.max(1)], DType::F32)?);
                }
            }
            "F" => held.clear(),
            other => return Err(format!("PROFILE_STEPS: unknown step {other:?} (E, S, A, O, F)")),
        }
        let cur = rt.current_allocated_bytes();
        let peak = rt.peak_allocated_bytes();
        cum_peak = cum_peak.max(peak);
        recs.push(format!(
            "{{\"n\":{n},\"step\":{},\"alloc_delta_mib\":{:.2},\"allocated_above_model_mib\":{:.2},\"peak_above_model_mib\":{:.2}}}",
            jstr(step),
            (cur as f64 - before as f64) / MIB,
            (cur as f64 - loaded as f64) / MIB,
            (peak as f64 - loaded as f64) / MIB
        ));
        rt.reset_peak_allocated_bytes();
    }
    println!(
        "{{\"workload\":{},\"operands\":{},\"rows\":{rows},\"async\":{},\"ple_run_bytes\":{ple_bytes},\"steps\":{},\"forwards\":{forwards},\"prelude_cached_mib\":{:.2},\"peak_above_model_mib\":{:.2},\"trace\":[{}]}}",
        jstr(&wl),
        jstr(if operands == GemmOperands::Bf16 { "bf16" } else { "f32" }),
        u8::from(async_encode),
        jstr(&steps),
        (prelude_cached as f64 - loaded as f64) / MIB,
        (cum_peak as f64 - loaded as f64) / MIB,
        recs.join(",")
    );
    Ok(())
}

fn mode_memory_sweep() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let out_path = std::env::var("PROFILE_JSON").unwrap_or_else(|_| "target/profile_memory.json".into());
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    // (label, PLE run bytes, steps, run the earlier workloads first, as
    // `bench_embedgemma2` does).
    let configs: Vec<(&str, Option<usize>, &str, bool)> = vec![
        ("bench-order", None, "E,E,E,E", true),
        ("alone-back-to-back", None, "E,E,E,E", false),
        ("alone-sync-between", None, "E,S,E,S,E", false),
        ("single", None, "E", false),
        ("paper-head", None, "A,F", false),
        ("paper-2a7cb8b", None, "O,F", false),
        ("ple64", Some(64 << 20), "E,E,S,E", false),
        ("ple16", Some(16 << 20), "E,E,S,E", false),
        ("ple-one-layer", Some(1), "E,E,S,E", false),
    ];
    let wls: Vec<&str> = spec.split(',').map(str::trim).collect();
    let mut results = Vec::new();
    for (wi, wl) in wls.iter().enumerate() {
        for &(label, ple, steps, ordered) in &configs {
            let mut cmd = std::process::Command::new(&exe);
            cmd.env("PROFILE_MODE", "memory")
                .env("PROFILE_WORKLOAD", wl)
                .env("PROFILE_STEPS", steps);
            if ordered && wi > 0 {
                cmd.env("PROFILE_PRELUDE", wls[..wi].join(","));
            } else {
                cmd.env_remove("PROFILE_PRELUDE");
            }
            if let Some(p) = ple {
                cmd.env("PROFILE_PLE_BYTES", p.to_string());
            } else {
                cmd.env_remove("PROFILE_PLE_BYTES");
            }
            let out = cmd.output().map_err(|e| format!("spawn: {e}"))?;
            if !out.status.success() {
                return Err(format!(
                    "{wl} {label}: child failed: {}",
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
            let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
            println!("{label:<20} {line}");
            results.push(format!("{{\"label\":{},\"result\":{line}}}", jstr(label)));
        }
    }
    write_json(
        &out_path,
        &format!("{{\"mode\":\"memory-sweep\",\"runs\":[{}]}}", results.join(",")),
    )?;
    Ok(())
}

/// Mode `memory-table`: the device peak above the loaded model of each
/// workload (`BENCH_WORKLOADS`) under both `GemmOperands`, one fresh child
/// process per cell, for one `encode`, for back-to-back `encode`s with no
/// wait between, and for `encode`s with an explicit `synchronize` between
/// (`single`, `back-to-back`, `sync-between`: the first, the last of four, and
/// the last of three). It uses only the library's public API, so the same
/// binary source measures a build before and after a memory change.
/// `PROFILE_JSON` names the output.
fn mode_memory_table() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let out_path = std::env::var("PROFILE_JSON").unwrap_or_else(|_| "target/profile_memory_table.json".into());
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    let configs: [(&str, &str); 3] = [
        ("single", "E"),
        ("back-to-back", "E,E,E,E"),
        ("sync-between", "E,S,E,S,E"),
    ];
    let mut results = Vec::new();
    for operands in ["f32", "bf16"] {
        for wl in spec.split(',').map(str::trim) {
            for (label, steps) in configs {
                let out = std::process::Command::new(&exe)
                    .env("PROFILE_MODE", "memory")
                    .env("PROFILE_WORKLOAD", wl)
                    .env("PROFILE_STEPS", steps)
                    .env("PROFILE_OPERANDS", operands)
                    .env_remove("PROFILE_PRELUDE")
                    .env_remove("PROFILE_PLE_BYTES")
                    .output()
                    .map_err(|e| format!("spawn: {e}"))?;
                if !out.status.success() {
                    return Err(format!(
                        "{operands} {wl} {label}: child failed: {}",
                        String::from_utf8_lossy(&out.stderr)
                    ));
                }
                let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
                println!("{operands:<5} {label:<13} {line}");
                results.push(format!(
                    "{{\"operands\":{},\"label\":{},\"result\":{line}}}",
                    jstr(operands),
                    jstr(label)
                ));
            }
        }
    }
    write_json(
        &out_path,
        &format!("{{\"mode\":\"memory-table\",\"runs\":[{}]}}", results.join(",")),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Mode: ledger
// ---------------------------------------------------------------------------

fn mode_ledger() -> Result<(), String> {
    let snap = snapshot()?;
    let wl = std::env::var("PROFILE_WORKLOAD").unwrap_or_else(|_| "8x1024".into());
    let (b, t) = parse_workload(&wl)?;
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    let weights = Weights::load(&rt, &st, &cfg)?;
    let rows = b * t;
    let seqs: Vec<Vec<u32>> = (0..b).map(|s| ids(s, t)).collect();
    let acts = ActsR::new(&rt, &cfg, b, rows, tessl::embedgemma2::PLE_RUN_BYTES, true)?;
    let io = IoR::new(&rt, &cfg, &seqs)?;
    rt.synchronize()?;
    let cx = Cx {
        rt: &rt,
        cfg: &cfg,
        w: &weights,
        a: &acts,
        io: &io,
        nb: b as u32,
        seq: t as u32,
        ledger: Some(RefCell::new(Ledger::default())),
    };
    let before = rt.current_allocated_bytes();
    forward(&cx, ALL)?;
    let l = cx.ledger.as_ref().expect("ledger set").borrow();
    println!("per-stage fresh device allocation during {wl} (the acts are already allocated; MiB):");
    let mut json = Vec::new();
    for (c, nm) in NAMES.iter().enumerate() {
        let (fresh, kept) = l.per_class[c];
        println!(
            "  {nm:<34} fresh {:>8.2}  kept {:>8.2}",
            fresh as f64 / MIB,
            kept as f64 / MIB
        );
        json.push(format!(
            "{{\"class\":{},\"fresh_mib\":{:.3},\"kept_mib\":{:.3}}}",
            jstr(nm),
            fresh as f64 / MIB,
            kept as f64 / MIB
        ));
    }
    println!(
        "net change in allocated bytes over the forward: {:.2} MiB",
        (rt.current_allocated_bytes() as f64 - before as f64) / MIB
    );
    if let Ok(path) = std::env::var("PROFILE_JSON") {
        write_json(
            &path,
            &format!(
                "{{\"mode\":\"ledger\",\"workload\":{},\"stages\":[{}]}}",
                jstr(&wl),
                json.join(",")
            ),
        )?;
    }
    Ok(())
}

/// The activation set itemized, HEAD against 2a7cb8b, with the bytes the pool
/// charges for each (`GpuRuntime::allocated_bytes_for`, Cold). Needs no GPU.
fn mode_layout() -> Result<(), String> {
    let snap = snapshot()?;
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let charged = |el: usize| GpuRuntime::allocated_bytes_for(el.max(1) * 4, BufferKind::Cold) as f64 / MIB;
    let mut json = Vec::new();
    for w in spec.split(',') {
        let (b, t) = parse_workload(w)?;
        let rows = b * t;
        let head = acts_layout(&cfg, b, rows, tessl::embedgemma2::PLE_RUN_BYTES);
        let old = acts_layout_2a7cb8b(&cfg, b, rows);
        println!("== {w} ({rows} rows): bytes the pool charges, MiB");
        println!("   {:<26} {:>9}   | {:<14} {:>9}", "HEAD", "MiB", "2a7cb8b", "MiB");
        for i in 0..head.len().max(old.len()) {
            let l = head
                .get(i)
                .map_or(String::new(), |(n, e)| format!("{n:<26} {:>9.2}", charged(*e)));
            let r = old
                .get(i)
                .map_or(String::new(), |(n, e)| format!("{n:<14} {:>9.2}", charged(*e)));
            println!("   {l:<36}   | {r}");
        }
        let (th, to): (f64, f64) = (
            head.iter().map(|&(_, e)| charged(e)).sum(),
            old.iter().map(|&(_, e)| charged(e)).sum(),
        );
        println!(
            "   total {th:>30.2}   | total {to:>22.2}  (HEAD / 2a7cb8b = {:.3})",
            th / to
        );
        let items = |v: &[(&str, usize)]| {
            v.iter()
                .map(|&(n, e)| {
                    format!(
                        "{{\"buffer\":{},\"elements\":{e},\"charged_mib\":{:.3}}}",
                        jstr(n),
                        charged(e)
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        json.push(format!(
            "{{\"workload\":{},\"rows\":{rows},\"head_total_mib\":{th:.3},\"2a7cb8b_total_mib\":{to:.3},\"head\":[{}],\"2a7cb8b\":[{}]}}",
            jstr(w),
            items(&head),
            items(&old)
        ));
    }
    if let Ok(path) = std::env::var("PROFILE_JSON") {
        write_json(
            &path,
            &format!("{{\"mode\":\"layout\",\"workloads\":[{}]}}", json.join(",")),
        )?;
    }
    Ok(())
}

/// What allocating `encode`'s activation set costs per call, without any
/// forward: allocate the set (`acts_layout`), queue one tiny dispatch, wait,
/// drop the set. The replica above keeps one set for all its forwards;
/// `encode` makes a new one per call, and its time exceeds the replica's by
/// more than the upload and readback can explain.
fn mode_alloc() -> Result<(), String> {
    let snap = snapshot()?;
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    let rounds = env_usize("PROFILE_ROUNDS", 9, 1)?;
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(true)?;
    let tiny = rt.alloc_buffer(64 * 4)?;
    let mut json = Vec::new();
    for w in spec.split(',') {
        let (b, t) = parse_workload(w)?;
        let rows = b * t;
        let lay = acts_layout(&cfg, b, rows, tessl::embedgemma2::PLE_RUN_BYTES);
        let alloc_set = || -> Result<Vec<Tensor>, String> {
            lay.iter()
                .map(|&(_, el)| rt.alloc_tensor_unzeroed(&[el.max(1)], DType::F32))
                .collect()
        };
        let mut results: Vec<(&str, Vec<f64>)> = Vec::new();
        for (label, sync_after_drop, reuse) in [
            ("alloc, dispatch, wait, drop (as encode)", false, false),
            ("alloc, dispatch, wait, drop, wait", true, false),
            ("one set kept: dispatch, wait", false, true),
        ] {
            let kept = if reuse { Some(alloc_set()?) } else { None };
            let mut samples = Vec::new();
            for r in 0..rounds + 3 {
                let (ms, _, _) = gated(|| {
                    let t0 = Instant::now();
                    let set = if reuse { None } else { Some(alloc_set()?) };
                    scale_f32_inplace(&rt, &tiny, 1.0, 64)?;
                    rt.synchronize()?;
                    drop(set);
                    if sync_after_drop {
                        rt.synchronize()?;
                    }
                    Ok(t0.elapsed().as_secs_f64() * 1e3)
                })?;
                if r >= 3 {
                    samples.push(ms);
                }
            }
            drop(kept);
            rt.synchronize()?;
            results.push((label, samples));
        }
        println!(
            "== {w} ({rows} rows, set {:.0} MiB)",
            lay.iter().map(|&(_, e)| e as f64 * 4.0).sum::<f64>() / MIB
        );
        for (label, s) in &results {
            println!(
                "   {label:<44} min {:>8.3} ms  median {:>8.3} ms",
                min_of(s),
                median_of(s)
            );
        }
        json.push(format!(
            "{{\"workload\":{},\"rows\":{rows},\"runs\":[{}]}}",
            jstr(w),
            results
                .iter()
                .map(|(l, s)| format!("{{\"what\":{},\"ms\":{}}}", jstr(l), jarr(s)))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    if let Ok(path) = std::env::var("PROFILE_JSON") {
        write_json(
            &path,
            &format!("{{\"mode\":\"alloc\",\"workloads\":[{}]}}", json.join(",")),
        )?;
    }
    Ok(())
}

/// What bounding the per-layer-input run buffer costs in time: `encode` timed
/// at several `set_ple_run_bytes`, interleaved, min of `PROFILE_ROUNDS`.
fn mode_ple() -> Result<(), String> {
    let snap = snapshot()?;
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| "32x256,8x1024".into());
    let rounds = env_usize("PROFILE_ROUNDS", 9, 1)?;
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(true)?;
    let mut model = EmbedGemma2Model::load(&rt, &st, PREFIX, cfg)?;
    let settings: [(&str, usize); 4] = [
        ("256 MiB (default)", tessl::embedgemma2::PLE_RUN_BYTES),
        ("64 MiB", 64 << 20),
        ("16 MiB", 16 << 20),
        ("one layer per run", 1),
    ];
    let mut json = Vec::new();
    for w in spec.split(',') {
        let (b, t) = parse_workload(w)?;
        let seqs: Vec<Vec<u32>> = (0..b).map(|s| ids(s, t)).collect();
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); settings.len()];
        let mut flags: Vec<Vec<bool>> = vec![Vec::new(); settings.len()];
        for &(_, bytes) in &settings {
            model.set_ple_run_bytes(bytes)?;
            model.encode(&batch, None, false)?;
        }
        for round in 0..rounds {
            for i in round_order(round, settings.len()) {
                model.set_ple_run_bytes(settings[i].1)?;
                let (ms, c, _) = gated(|| {
                    let t0 = Instant::now();
                    model.encode(&batch, None, false)?;
                    Ok(t0.elapsed().as_secs_f64() * 1e3)
                })?;
                samples[i].push(ms);
                flags[i].push(c);
            }
        }
        println!("== {w} ({} rows): encode ms by PLE run buffer", b * t);
        let base = min_of(&clean_or_all(&samples[0], &flags[0]));
        for (i, (label, _)) in settings.iter().enumerate() {
            let s = clean_or_all(&samples[i], &flags[i]);
            println!(
                "   {label:<20} min {:>9.3}  median {:>9.3}  ({:+.1}% vs default)",
                min_of(&s),
                median_of(&s),
                100.0 * (min_of(&s) / base - 1.0)
            );
        }
        json.push(format!(
            "{{\"workload\":{},\"runs\":[{}]}}",
            jstr(w),
            settings
                .iter()
                .enumerate()
                .map(|(i, (l, bytes))| format!(
                    "{{\"ple_run_bytes\":{bytes},\"label\":{},\"ms\":{},\"contended\":{}}}",
                    jstr(l),
                    jarr(&samples[i]),
                    jbools(&flags[i])
                ))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    if let Ok(path) = std::env::var("PROFILE_JSON") {
        write_json(
            &path,
            &format!("{{\"mode\":\"ple\",\"workloads\":[{}]}}", json.join(",")),
        )?;
    }
    Ok(())
}

/// FNV-1a over the bit patterns of `v`: two runs agree bit for bit exactly
/// when their hashes do (up to a 2^-64 collision).
fn fnv1a_bits(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
    })
}

/// Mode `hash`: the embeddings of each workload (`BENCH_WORKLOADS`; `+` joins
/// uniform `BxT` terms into one ragged batch) under both `GemmOperands`, each as
/// an FNV-1a hash of the f32 bit patterns, from three consecutive `encode`
/// calls (the second and third must reproduce the first, bit for bit) and one
/// `truncate_dim` 128 call. The tool for "an optimisation changed no output
/// bit": run it on two builds and `diff` the lines.
fn mode_hash() -> Result<(), String> {
    let snap = snapshot()?;
    let spec =
        std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| "1x16,64x32,32x256,8x1024,2x4096,1x4096+63x32".into());
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    for (label, operands) in [("f32", GemmOperands::ExactF32), ("bf16", GemmOperands::Bf16)] {
        let mut model = EmbedGemma2Model::load_with(&rt, &st, PREFIX, cfg.clone(), operands)?;
        if let Ok(v) = std::env::var("PROFILE_PLE_BYTES") {
            let n: usize = v.parse().map_err(|e| format!("PROFILE_PLE_BYTES={v:?}: {e}"))?;
            model.set_ple_run_bytes(n)?;
        }
        for wl in spec.split(',').map(str::trim) {
            let mut seqs: Vec<Vec<u32>> = Vec::new();
            for term in wl.split('+') {
                let (b, t) = parse_workload(term)?;
                seqs.extend((0..b).map(|s| ids(s, t)));
            }
            let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
            let runs: Vec<u64> = (0..3)
                .map(|_| model.encode(&batch, None, false).map(|o| fnv1a_bits(&o.embeddings)))
                .collect::<Result<_, _>>()?;
            let trunc = fnv1a_bits(&model.encode(&batch, Some(128), false)?.embeddings);
            let stable = runs.iter().all(|&h| h == runs[0]);
            println!(
                "HASH {label} {wl} {:016x} trunc128 {trunc:016x} repeat_stable={stable}",
                runs[0]
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    match std::env::var("PROFILE_MODE").as_deref() {
        Ok("hash") => mode_hash(),
        Ok("ple") => mode_ple(),
        Ok("alloc") => mode_alloc(),
        Ok("layout") => mode_layout(),
        Err(_) | Ok("stages") => mode_stages(),
        Ok("gemm") => mode_gemm(),
        Ok("memory") => mode_memory(),
        Ok("memory-sweep") => mode_memory_sweep(),
        Ok("memory-table") => mode_memory_table(),
        Ok("ledger") => mode_ledger(),
        Ok(other) => Err(format!(
            "PROFILE_MODE={other:?}: expected stages, gemm, memory, memory-sweep, memory-table, ledger, layout, alloc, ple or hash"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{pairs, parse_workload};

    #[test]
    fn workload_grammar() {
        assert_eq!(parse_workload("8x1024").unwrap(), (8, 1024));
        for bad in ["0x5", "2x1", "2", "x5", "2x5+1"] {
            assert!(parse_workload(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn window_pairs() {
        // Window wider than the sequence: every pair.
        assert_eq!(pairs(10, 0), 100.0);
        assert_eq!(pairs(10, 20), 100.0);
        // seq 4, |i-j| <= 1: 2+3+3+2.
        assert_eq!(pairs(4, 1), 10.0);
    }
}
