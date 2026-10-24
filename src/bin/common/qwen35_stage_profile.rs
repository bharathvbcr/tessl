//! `bench_qwen35_layers --stage-profile`: where one Qwen3.5-2B prefill+score
//! spends its time, stage by stage, on the work the head-to-head lane times.
//!
//! The model has no stage seam below `LayerPart`, so the stages are timed on
//! a replica of one GDN layer, one attention layer and one MLP built from the
//! same public ops in the same order and dtypes as `Qwen35Model::gdn`,
//! `::attention` and `::mlp` (`src/qwen35_model.rs`), at the 2B shapes, with
//! random bf16 weights. Every stage runs `REPS` times per command buffer under
//! async encode, as `time_parts` does, so the submit floor is not counted per
//! stage. The replica is cross-checked against the real model: the same run
//! times `Staged::encode_part` for the layers' parts (real checkpoint, real
//! ids) and the stage sums must reproduce them, or the table is not trusted.
//! The whole forward is then split on the host timeline (begin, the encode
//! return of the 24 layers, the wait in score_answers) to expose what is not
//! GPU kernel time.
//!
//! Output: a readable table and one `STAGE_PROFILE_JSON` line per T,
//! `bench/qwen35_stage_profile_report.py` merges it with the MLX lane's
//! `bench/qwen35_prefill_mlx_profile.py`.

use std::sync::Arc;
use std::time::Instant;

use tessl::nn::AttnDims;
use tessl::qwen35::{self, Cols, GdnChunkPhase, GdnParams, OutCols, StateIn};
use tessl::qwen35_model::{LayerKind, Qwen35Model};
use tessl::tensor::GpuBuffer;
use tessl::{gemm, DType, GpuRuntime, Tensor};

use super::prefill_ids::PrefillIds;
use super::{bounded, buf, common, finite_nonzero, time_parts, weight, weight_bits, Res, BACKEND, WARMUP};

/// Commands per command buffer for a stage: the same `REPS` as `time_parts`.
const STAGE_REPS: usize = 8;

/// One timed stage of the replica.
struct Stage {
    group: &'static str,
    name: &'static str,
    /// Median and minimum milliseconds of one call.
    ms: f64,
    min_ms: f64,
    /// Dispatches one call encodes.
    dispatches: usize,
    /// Host microseconds to encode one call (no waiting).
    enc_us: f64,
    /// How many times a forward runs this stage.
    count: usize,
}

/// Median and min milliseconds per call of `f`, `STAGE_REPS` per command
/// buffer, plus the dispatches of one call and the host encode time per call.
fn time_stage(rt: &Arc<GpuRuntime>, iters: usize, mut f: impl FnMut() -> Res<()>) -> Res<(f64, f64, usize, f64)> {
    rt.synchronize()?;
    rt.take_dispatch_count();
    f()?;
    let dispatches = rt.take_dispatch_count();
    rt.synchronize()?;
    for _ in 0..WARMUP {
        for _ in 0..STAGE_REPS {
            f()?;
        }
        rt.synchronize()?;
    }
    let (mut samples, mut enc) = (Vec::with_capacity(iters), Vec::with_capacity(iters));
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..STAGE_REPS {
            f()?;
        }
        let issued = t0.elapsed();
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3 / STAGE_REPS as f64);
        enc.push(issued.as_secs_f64() * 1e6 / STAGE_REPS as f64);
    }
    let min = common::min_of(&samples);
    Ok((common::median(samples)?, min, dispatches, common::median(enc)?))
}

/// The replica's buffers at `t` tokens: `qwen35_model::Acts`, minus the
/// fields no stage here reads.
struct Replica {
    t: u32,
    resid: Tensor,
    x: Tensor,
    g_proj: Tensor,
    g_qkv: GpuBuffer,
    g_o: GpuBuffer,
    g_y: Tensor,
    g_ws: qwen35::GdnWorkspace,
    g_state: GpuBuffer,
    g_conv_state: GpuBuffer,
    a_proj: Tensor,
    a_q: GpuBuffer,
    a_k: GpuBuffer,
    a_v: GpuBuffer,
    a_o: GpuBuffer,
    a_y: Tensor,
    tkv: GpuBuffer,
    zero: GpuBuffer,
    m_gate: Tensor,
    m_up: Tensor,
    m_mid: Tensor,
    ids: GpuBuffer,
    embed_table: GpuBuffer,
    // Weights (shapes of layer 0 / 3, random values).
    input_norm: GpuBuffer,
    g_w_in: Tensor,
    g_conv_w: GpuBuffer,
    g_a_log: GpuBuffer,
    g_dt_bias: GpuBuffer,
    g_norm_w: GpuBuffer,
    g_w_out: Tensor,
    a_w_in: Tensor,
    a_q_norm: GpuBuffer,
    a_k_norm: GpuBuffer,
    a_w_out: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
}

/// Rows of the stand-in embedding table: the gather reads `t` rows of it,
/// so the cost does not depend on the 248k-row vocabulary.
const TABLE_ROWS: usize = 4096;

impl Replica {
    fn new(rt: &Arc<GpuRuntime>, model: &Qwen35Model, t: usize) -> Res<Self> {
        let cfg = model.config();
        let (h, inter) = (cfg.hidden as usize, cfg.intermediate as usize);
        let (g, l) = (cfg.gdn, cfg.attn);
        let t32 = t as u32;
        let dims = g.dims(1, t32);
        let f32t = |cols: usize| rt.alloc_tensor_f32(&[t, cols]);
        let bf = |cols: usize| rt.alloc_tensor_bf16(&[t, cols]);
        let f32b = |n: usize| rt.alloc_buffer(n.max(1) * 4);
        let qd = (l.q_heads() * l.head_dim()) as usize;
        let kvd = (l.kv_heads() * l.head_dim()) as usize;
        let vh = g.v_heads() as usize;
        let mut seed = 1000u64;
        let mut w = |rows: usize, cols: usize| {
            seed += 1;
            weight(rt, rows, cols, &weight_bits(rows, cols, seed))
        };
        let resid = f32t(h)?;
        resid.buffer.write_f32(&common::fill_affine(t * h, 7, 1.0, 0.0));
        let ids: Vec<u32> = (0..t).map(|i| (i * 7919 % TABLE_ROWS) as u32).collect();
        Ok(Self {
            t: t32,
            resid,
            x: bf(h)?,
            g_proj: f32t(g.width() as usize)?,
            g_qkv: f32b(t * g.conv_dim() as usize)?,
            g_o: f32b(t * g.value_dim() as usize)?,
            g_y: bf(g.value_dim() as usize)?,
            g_ws: qwen35::GdnWorkspace::new(rt, &dims)?,
            g_state: f32b(dims.state_elems_per_row())?,
            g_conv_state: f32b(g.conv_dim() as usize * (cfg.conv_kernel as usize - 1))?,
            a_proj: f32t(l.width() as usize)?,
            a_q: f32b(t * qd)?,
            a_k: f32b(t * kvd)?,
            a_v: f32b(t * kvd)?,
            a_o: f32b(t * qd)?,
            a_y: bf(qd)?,
            tkv: rt.alloc_buffer_from_u32(&[t32])?,
            zero: rt.alloc_buffer_from_u32(&[0])?,
            m_gate: f32t(inter)?,
            m_up: f32t(inter)?,
            m_mid: bf(inter)?,
            ids: rt.alloc_buffer_from_u32(&ids)?,
            embed_table: {
                let b = rt.alloc_buffer(TABLE_ROWS * h * 2)?;
                b.write_bf16_bits(&tessl::tensor::f32_slice_to_bf16(&common::fill_affine(
                    TABLE_ROWS * h,
                    5,
                    0.5,
                    0.0,
                )));
                b
            },
            input_norm: bounded(rt, h, 11, 0.3)?,
            g_w_in: w(h, g.width() as usize)?,
            g_conv_w: bounded(rt, g.conv_dim() as usize * cfg.conv_kernel as usize, 12, 0.5)?,
            g_a_log: buf(
                rt,
                &common::fill_affine(vh, 33, 1.0, 0.0)
                    .iter()
                    .map(|v| 1.4 + 1.3 * v)
                    .collect::<Vec<_>>(),
            )?,
            g_dt_bias: buf(
                rt,
                &common::fill_affine(vh, 34, 1.0, 0.0)
                    .iter()
                    .map(|v| -4.5 + 2.5 * v)
                    .collect::<Vec<_>>(),
            )?,
            g_norm_w: bounded(rt, g.v_dim() as usize, 13, 0.3)?,
            g_w_out: w(g.value_dim() as usize, h)?,
            a_w_in: w(h, l.width() as usize)?,
            a_q_norm: bounded(rt, l.head_dim() as usize, 14, 0.3)?,
            a_k_norm: bounded(rt, l.head_dim() as usize, 15, 0.3)?,
            a_w_out: w(qd, h)?,
            gate_w: w(h, inter)?,
            up_w: w(h, inter)?,
            down_w: w(inter, h)?,
        })
    }
}

/// Run every stage of a GDN layer, an attention layer and an MLP once, in
/// order, so each stage's input holds what the stage before it wrote. Also
/// the body `time_stage` repeats per stage.
struct Ops<'a> {
    rt: &'a Arc<GpuRuntime>,
    m: &'a Qwen35Model,
    r: &'a Replica,
}

impl Ops<'_> {
    fn norm(&self, w: &GpuBuffer) -> Res<()> {
        let cfg = self.m.config();
        qwen35::rms_norm(
            self.rt,
            &self.r.resid.buffer,
            w,
            &self.r.x.buffer,
            DType::BF16,
            self.r.t,
            cfg.hidden,
            cfg.rms_norm_eps,
        )
    }

    fn embed(&self) -> Res<()> {
        let cfg = self.m.config();
        qwen35::embed_rows(
            self.rt,
            &self.r.ids,
            self.r.t,
            qwen35::LmHead {
                weight: &self.r.embed_table,
                dtype: DType::BF16,
                vocab: TABLE_ROWS as u32,
            },
            cfg.hidden,
            &self.r.resid.buffer,
        )
    }

    fn g_in_proj(&self) -> Res<()> {
        gemm(&self.r.x, &self.r.g_w_in, &self.r.g_proj, BACKEND)
    }

    fn g_conv(&self) -> Res<()> {
        let (cfg, r) = (self.m.config(), self.r);
        let g = cfg.gdn;
        qwen35::conv1d_silu(
            self.rt,
            Cols::dense(&r.g_proj.buffer, g.width()),
            &r.g_conv_w,
            cfg.conv_kernel,
            StateIn::Zero,
            &r.g_qkv,
            Some(&r.g_conv_state),
            1,
            r.t,
            g.conv_dim(),
        )
    }

    fn g_scan(&self, phase: Option<GdnChunkPhase>) -> Res<()> {
        let (cfg, r) = (self.m.config(), self.r);
        let g = cfg.gdn;
        let dims = g.dims(1, r.t);
        let (qkv, gates) = (g.conv_qkv(&r.g_qkv), g.gates(&r.g_proj.buffer));
        let params = GdnParams {
            a_log: &r.g_a_log,
            dt_bias: &r.g_dt_bias,
        };
        let out = Cols::dense(&r.g_o, g.value_dim());
        match phase {
            None => qwen35::gdn_chunk_forward(
                self.rt,
                &dims,
                &qkv,
                &gates,
                &params,
                StateIn::Zero,
                &r.g_ws,
                out,
                Some(&r.g_state),
            ),
            Some(p) => qwen35::gdn_chunk_phase(
                self.rt,
                &dims,
                &qkv,
                &gates,
                &params,
                StateIn::Zero,
                &r.g_ws,
                out,
                Some(&r.g_state),
                p,
            ),
        }
    }

    fn g_gated_norm(&self) -> Res<()> {
        let (cfg, r) = (self.m.config(), self.r);
        let g = cfg.gdn;
        qwen35::gated_rms_norm(
            self.rt,
            Cols::dense(&r.g_o, g.value_dim()),
            Cols {
                buf: &r.g_proj.buffer,
                ld: g.width(),
                off: g.z_off(),
            },
            &r.g_norm_w,
            OutCols {
                cols: Cols::dense(&r.g_y.buffer, g.value_dim()),
                dtype: DType::BF16,
            },
            r.t,
            g.v_heads(),
            g.v_dim(),
            cfg.rms_norm_eps,
        )
    }

    fn g_out(&self) -> Res<()> {
        qwen35::project_residual(&self.r.g_y, &self.r.g_w_out, &self.r.resid, BACKEND)
    }

    fn a_in_proj(&self) -> Res<()> {
        gemm(&self.r.x, &self.r.a_w_in, &self.r.a_proj, BACKEND)
    }

    fn a_shape(&self) -> qwen35::AttnShape {
        let (cfg, l) = (self.m.config(), self.m.config().attn);
        qwen35::AttnShape {
            batch: 1,
            seq: self.r.t,
            q_heads: l.q_heads(),
            kv_heads: l.kv_heads(),
            head_dim: l.head_dim(),
            rotary_dim: cfg.rotary_dim,
        }
    }

    fn a_qk_rope(&self) -> Res<()> {
        let (cfg, r) = (self.m.config(), self.r);
        qwen35::attn_qk_norm_rope(
            self.rt,
            &self.a_shape(),
            Cols::dense(&r.a_proj.buffer, cfg.attn.width()),
            &r.a_q_norm,
            &r.a_k_norm,
            &qwen35::AttnTargets {
                q_out: &r.a_q,
                k_cache: &r.a_k,
                v_cache: &r.a_v,
            },
            0,
            cfg.rope_theta,
            cfg.rms_norm_eps,
        )
    }

    fn a_sdpa(&self) -> Res<()> {
        let (l, r) = (self.m.config().attn, self.r);
        let dims = AttnDims {
            batch: 1,
            tq: r.t,
            heads: l.q_heads(),
            heads_kv: l.kv_heads(),
            window: 0,
            scale: 1.0 / (l.head_dim() as f32).sqrt(),
        };
        qwen35::attn_prefill(
            self.rt, &r.a_q, &r.a_k, &r.a_v, &r.a_o, &r.tkv, &r.zero, &r.zero, dims, false,
        )
    }

    fn a_gate(&self) -> Res<()> {
        let (l, r) = (self.m.config().attn, self.r);
        let width = l.q_heads() * l.head_dim();
        qwen35::attn_output_gate(
            self.rt,
            &r.a_o,
            Cols::dense(&r.a_proj.buffer, l.width()),
            OutCols {
                cols: Cols::dense(&r.a_y.buffer, width),
                dtype: DType::BF16,
            },
            r.t,
            l.q_heads(),
            l.head_dim(),
        )
    }

    fn a_out(&self) -> Res<()> {
        qwen35::project_residual(&self.r.a_y, &self.r.a_w_out, &self.r.resid, BACKEND)
    }

    fn mlp_gate_up(&self) -> Res<()> {
        gemm(&self.r.x, &self.r.gate_w, &self.r.m_gate, BACKEND)?;
        gemm(&self.r.x, &self.r.up_w, &self.r.m_up, BACKEND)
    }

    fn mlp_swiglu(&self) -> Res<()> {
        let (inter, r) = (self.m.config().intermediate, self.r);
        qwen35::swiglu(
            self.rt,
            Cols::dense(&r.m_gate.buffer, inter),
            Cols::dense(&r.m_up.buffer, inter),
            OutCols {
                cols: Cols::dense(&r.m_mid.buffer, inter),
                dtype: DType::BF16,
            },
            r.t,
            inter,
        )
    }

    fn mlp_down(&self) -> Res<()> {
        qwen35::project_residual(&self.r.m_mid, &self.r.down_w, &self.r.resid, BACKEND)
    }
}

/// One host-timeline sample of the real forward at `t`.
struct Timeline {
    begin: f64,
    advance_return: f64,
    score_wait: f64,
    total: f64,
}

fn timeline(model: &Qwen35Model, ids: &[u32], answers: &[u32]) -> Res<Timeline> {
    let layers = model.config().layers.len();
    let t0 = Instant::now();
    let mut s = model.begin(ids)?;
    let t1 = Instant::now();
    s.advance_to(layers)?;
    let t2 = Instant::now();
    s.score_answers(&[(ids.len() - 1) as u32], answers)?;
    let t3 = Instant::now();
    let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1e3;
    Ok(Timeline {
        begin: ms(t0, t1),
        advance_return: ms(t1, t2),
        score_wait: ms(t2, t3),
        total: ms(t0, t3),
    })
}

pub fn run(
    rt: &Arc<GpuRuntime>,
    model: &Qwen35Model,
    ts: &[usize],
    iters: usize,
    ids_path: &std::path::Path,
) -> Res<()> {
    let file = PrefillIds::load(ids_path)?;
    if file.vocab != model.config().vocab {
        return Err(format!(
            "ids file vocab {} but the model's is {}",
            file.vocab,
            model.config().vocab
        ));
    }
    let kinds = &model.config().layers;
    let n_gdn = kinds.iter().filter(|&&k| k == LayerKind::LinearAttention).count();
    let n_attn = kinds.len() - n_gdn;
    println!(
        "stage profile: {n_gdn} GDN + {n_attn} attention layers, {STAGE_REPS} calls per command buffer, {iters} timed samples (median; min in the JSON), async encode"
    );
    println!(
        "host: load/GPU at start: GPU utilization {}%",
        common::gpu_utilization()
    );
    rt.set_async_encode(true)?;
    for &t in ts {
        let ids = file.prefix(t).map_err(|e| format!("T={t}: {e}"))?;
        println!();
        println!(
            "=== T = {t} (ids {}, fnv1a32 {}) ===",
            ids_path.display(),
            super::prefill_ids::fnv1a32(ids)
        );
        println!("GPU utilization at start of T: {}%", common::gpu_utilization());
        let r = Replica::new(rt, model, t)?;
        let ops = Ops { rt, m: model, r: &r };
        // One pass in forward order so every stage's input is what the stage
        // before it wrote; then each stage alone.
        ops.embed()?;
        ops.norm(&r.input_norm)?;
        ops.g_in_proj()?;
        ops.g_conv()?;
        ops.g_scan(None)?;
        ops.g_gated_norm()?;
        ops.g_out()?;
        ops.a_in_proj()?;
        ops.a_qk_rope()?;
        ops.a_sdpa()?;
        ops.a_gate()?;
        ops.a_out()?;
        ops.mlp_gate_up()?;
        ops.mlp_swiglu()?;
        ops.mlp_down()?;
        rt.synchronize()?;

        let mut stages: Vec<Stage> = Vec::new();
        let mut add =
            |group: &'static str, name: &'static str, count: usize, f: &mut dyn FnMut() -> Res<()>| -> Res<()> {
                let (ms, min_ms, dispatches, enc_us) = time_stage(rt, iters, &mut *f)?;
                stages.push(Stage {
                    group,
                    name,
                    ms,
                    min_ms,
                    dispatches,
                    enc_us,
                    count,
                });
                Ok(())
            };
        add("embed", "gather", 1, &mut || ops.embed())?;
        add("gdn", "input_norm", n_gdn, &mut || ops.norm(&r.input_norm))?;
        add("gdn", "in_proj_gemm", n_gdn, &mut || ops.g_in_proj())?;
        add("gdn", "conv1d_silu", n_gdn, &mut || ops.g_conv())?;
        add("gdn", "gdn_core_prep", n_gdn, &mut || {
            ops.g_scan(Some(GdnChunkPhase::Prep))
        })?;
        add("gdn", "gdn_core_scan", n_gdn, &mut || {
            ops.g_scan(Some(GdnChunkPhase::Scan))
        })?;
        add("gdn", "gdn_core", n_gdn, &mut || ops.g_scan(None))?;
        add("gdn", "gated_norm", n_gdn, &mut || ops.g_gated_norm())?;
        add("gdn", "out_proj_resid", n_gdn, &mut || ops.g_out())?;
        add("attn", "input_norm", n_attn, &mut || ops.norm(&r.input_norm))?;
        add("attn", "in_proj_gemm", n_attn, &mut || ops.a_in_proj())?;
        add("attn", "qk_norm_rope", n_attn, &mut || ops.a_qk_rope())?;
        add("attn", "sdpa", n_attn, &mut || ops.a_sdpa())?;
        add("attn", "out_gate", n_attn, &mut || ops.a_gate())?;
        add("attn", "out_proj_resid", n_attn, &mut || ops.a_out())?;
        add("mlp", "post_norm", kinds.len(), &mut || ops.norm(&r.input_norm))?;
        add("mlp", "gate_up_gemm", kinds.len(), &mut || ops.mlp_gate_up())?;
        add("mlp", "swiglu", kinds.len(), &mut || ops.mlp_swiglu())?;
        add("mlp", "down_proj_resid", kinds.len(), &mut || ops.mlp_down())?;
        finite_nonzero(
            "replica residual after the stage loops",
            &r.resid,
            t * model.config().hidden as usize,
        )?;

        // The real model's parts, for the cross-check.
        let parts = time_parts(rt, model, ids, iters)?;

        // The forward on the host timeline, and the submit floor.
        let answers = &file.answers;
        for _ in 0..WARMUP {
            timeline(model, ids, answers)?;
        }
        let mut tl: Vec<Timeline> = (0..iters).map(|_| timeline(model, ids, answers)).collect::<Res<_>>()?;
        tl.sort_by(|a, b| a.total.total_cmp(&b.total));
        let best = &tl[0];
        let med = &tl[tl.len() / 2];
        // An empty submit-and-wait: the cost every command buffer pays.
        let floor = {
            let mut v = Vec::new();
            for _ in 0..iters.max(9) {
                let t0 = Instant::now();
                rt.synchronize()?;
                v.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            common::median(v)?
        };

        // Report.
        let sum = |g: &str| -> f64 {
            stages
                .iter()
                .filter(|s| s.group == g && s.name != "gdn_core_prep" && s.name != "gdn_core_scan")
                .map(|s| s.ms * s.count as f64)
                .sum()
        };
        println!(
            "{:<6} {:<16} {:>10} {:>6} {:>5} {:>10} {:>9}",
            "group", "stage", "ms/call", "count", "disp", "total ms", "enc us"
        );
        for s in &stages {
            let note = if s.name == "gdn_core_prep" || s.name == "gdn_core_scan" {
                " (split of gdn_core)"
            } else {
                ""
            };
            println!(
                "{:<6} {:<16} {:>10.4} {:>6} {:>5} {:>10.3} {:>9.1}{note}",
                s.group,
                s.name,
                s.ms,
                s.count,
                s.dispatches,
                s.ms * s.count as f64,
                s.enc_us
            );
        }
        let stage_total = sum("embed") + sum("gdn") + sum("attn") + sum("mlp");
        println!("sum of stages x counts (no final norm / score / sync): {stage_total:.3} ms");
        let gdn_stage_part = |names: &[&str]| -> f64 {
            stages
                .iter()
                .filter(|s| s.group == "gdn" && names.contains(&s.name))
                .map(|s| s.ms)
                .sum()
        };
        let attn_stage_part = |names: &[&str]| -> f64 {
            stages
                .iter()
                .filter(|s| s.group == "attn" && names.contains(&s.name))
                .map(|s| s.ms)
                .sum()
        };
        let mlp_stage: f64 = stages.iter().filter(|s| s.group == "mlp").map(|s| s.ms).sum();
        let gdn_stage = gdn_stage_part(&[
            "input_norm",
            "in_proj_gemm",
            "conv1d_silu",
            "gdn_core",
            "gated_norm",
            "out_proj_resid",
        ]);
        let attn_stage = attn_stage_part(&[
            "input_norm",
            "in_proj_gemm",
            "qk_norm_rope",
            "sdpa",
            "out_gate",
            "out_proj_resid",
        ]);
        println!("cross-check against the real model's parts (encode_part, real weights), ms per layer:");
        println!(
            "  gdn layer  (norm+mixer) real {:.4} vs replica stages {:.4}",
            parts.gdn_layer, gdn_stage
        );
        println!(
            "  attn layer (norm+mixer) real {:.4} vs replica stages {:.4}",
            parts.attn_layer, attn_stage
        );
        println!(
            "  mlp        (norm+mlp)   real {:.4} vs replica stages {:.4}",
            parts.mlp, mlp_stage
        );
        let parts_total =
            parts.gdn_layer * n_gdn as f64 + parts.attn_layer * n_attn as f64 + parts.mlp * kinds.len() as f64;
        println!("  real parts x layers {parts_total:.3} ms");
        println!(
            "host timeline of the real forward (min-total sample / median-total sample): begin {:.3}/{:.3} ms, advance_to(24) returns {:.3}/{:.3} ms, score_answers waits {:.3}/{:.3} ms, total {:.3}/{:.3} ms",
            best.begin, med.begin, best.advance_return, med.advance_return, best.score_wait, med.score_wait, best.total, med.total
        );
        println!("empty submit-and-wait floor (synchronize with nothing encoded): {floor:.4} ms");
        println!(
            "unaccounted = total {:.3} - real parts x layers {:.3} = {:.3} ms (begin allocs + gather, final norm + score, submit)",
            best.total,
            parts_total,
            best.total - parts_total
        );
        let rows: Vec<String> = stages
            .iter()
            .map(|s| {
                format!(
                    "{{\"group\":\"{}\",\"stage\":\"{}\",\"ms\":{},\"min_ms\":{},\"count\":{},\"dispatches\":{},\"enc_us\":{}}}",
                    s.group,
                    s.name,
                    common::jnum(s.ms),
                    common::jnum(s.min_ms),
                    s.count,
                    s.dispatches,
                    common::jnum(s.enc_us)
                )
            })
            .collect();
        println!(
            "STAGE_PROFILE_JSON {{\"backend\":\"tessl\",\"tokens\":{t},\"iters\":{iters},\"reps\":{STAGE_REPS},\"stages\":[{}],\"parts_ms\":{{\"gdn_layer\":{},\"attn_layer\":{},\"mlp\":{}}},\"timeline_ms\":{{\"begin\":{},\"advance_return\":{},\"score_wait\":{},\"total\":{},\"total_median\":{}}},\"submit_floor_ms\":{}}}",
            rows.join(","),
            common::jnum(parts.gdn_layer),
            common::jnum(parts.attn_layer),
            common::jnum(parts.mlp),
            common::jnum(best.begin),
            common::jnum(best.advance_return),
            common::jnum(best.score_wait),
            common::jnum(best.total),
            common::jnum(med.total),
            common::jnum(floor)
        );
    }
    rt.set_async_encode(false)?;
    Ok(())
}
