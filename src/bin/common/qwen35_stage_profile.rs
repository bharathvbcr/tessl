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
//! ids) and the stage sums must reproduce them: each of the gdn, attn and mlp
//! rows must agree within `REPLICA_REL_TOL` (5%, `|replica - real| <= tol *
//! real`), or `run` returns an error from `check_replica_against_parts` after
//! printing the human-readable table and cross-check lines and before the
//! `STAGE_PROFILE_JSON` line, so a failed run keeps the human evidence and emits no machine-readable
//! result. A non-finite or non-positive time is an error, never a pass.
//! Every requested T is measured even when an earlier one fails the cross-check
//! (`profile_every_t`): a passing T prints its `STAGE_PROFILE_JSON`, a failing T
//! prints none, and the run then prints one `STAGE_PROFILE_INCOMPLETE` line (the
//! requested and failed T values, plus `aborted_at` and `unmeasured` after a hard error) and returns one error naming every failing T,
//! so the exit is nonzero and a partial log is never read as a full table.
//! Before any timing for a T, an untimed kernel-set check compares which
//! kernels the replica's stages request against which the real model's
//! `encode_part` requests for the same layer (the first GDN layer for the gdn
//! and mlp groups, the first attention layer for attn, as `time_parts` picks
//! them). It reads the runtime's kernel trace (`TESSL_KERNEL_TRACE`, read once
//! per process: this module never sets it): every `GpuRuntime::pipeline`
//! request counts, cache hit or not, so the names whose count rose around a
//! side's work are the kernels it requested. A group whose delta is empty on
//! either side cannot be attested and is an error, never a pass; a missing or
//! extra kernel is a hard error naming it, raised before the T is timed (so
//! structural drift wastes no GPU time) through `profile_every_t`'s abort
//! path. With the trace unset the run prints one
//! `kernel-set check: NOT RUN (TESSL_KERNEL_TRACE unset)` line and goes on;
//! otherwise each T prints `kernel-set check: PASS T=<t>` or aborts.
//! The sdpa stage mirrors `Qwen35Model::prefill_attention` for the model's
//! precision: under bf16 with the 256-wide head the bf16 tile needs, it
//! includes the three f32-to-bf16 Q/K/V casts (as the real attention part
//! does) before the `Q32K64Sg4` bf16 tile; otherwise it is the f32 kernel.
//! The async-encode flag is restored to what the caller had on every path.
//! The whole forward is then split on the host timeline (begin, the encode
//! return of the 24 layers, the wait in score_answers) to expose what is not
//! GPU kernel time.
//!
//! Output: a readable table and one `STAGE_PROFILE_JSON` line per T,
//! `bench/qwen35_stage_profile_report.py` merges it with the MLX lane's
//! `bench/qwen35_prefill_mlx_profile.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use tessl::attn_prefill_bf16::{self, AttnProbTile};
use tessl::nn::AttnDims;
use tessl::qwen35::{self, AttnTile, Cols, GdnChunkPhase, GdnParams, OutCols, StateIn};
use tessl::qwen35_model::{LayerKind, LayerPart, Precision, Qwen35Model};
use tessl::runtime::{kernel_trace_enabled, traced_kernel_count, traced_kernels};
use tessl::tensor::GpuBuffer;
use tessl::{gemm, DType, GpuRuntime, Tensor};

use super::prefill_ids::PrefillIds;
use super::{bounded, buf, common, finite_nonzero, time_parts, weight, weight_bits, Res, BACKEND, WARMUP};

/// Commands per command buffer for a stage: the same `REPS` as `time_parts`.
const STAGE_REPS: usize = 8;

/// Largest relative gap `|replica - real| / real` a cross-check row may have.
const REPLICA_REL_TOL: f64 = 0.05;

/// Fail unless every `(name, real, replica)` row agrees within `rel_tol`.
/// Every failing row is named with its real, replica and relative gap. A row
/// that cannot be compared (non-finite, or `real <= 0`) fails, and so does an
/// empty table or a `rel_tol` outside `[0, 1)` (including NaN and infinity).
fn check_replica_against_parts(rows: &[(&str, f64, f64)], rel_tol: f64) -> Res<()> {
    // A NaN or infinite tolerance makes `gap > tol * real` false for every row, and
    // a tolerance of 1 or more lets a zero replica pass: neither may disable the check.
    if !rel_tol.is_finite() || !(0.0..1.0).contains(&rel_tol) {
        return Err(format!(
            "replica cross-check: tolerance {rel_tol} is invalid (need a finite value in [0, 1))"
        ));
    }
    if rows.is_empty() {
        return Err("replica cross-check: no rows to compare".to_string());
    }
    let mut failures = Vec::new();
    for &(name, real, replica) in rows {
        if !real.is_finite() || !replica.is_finite() || real <= 0.0 {
            failures.push(format!(
                "{name}: real {real} ms vs replica {replica} ms is not comparable (need finite times and real > 0)"
            ));
        } else if (replica - real).abs() > rel_tol * real {
            failures.push(format!(
                "{name}: real {real} ms vs replica {replica} ms, gap {:.2}% > {:.2}%",
                (replica - real).abs() / real * 100.0,
                rel_tol * 100.0
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "replica stages do not reproduce the real model's parts, so the stage table is not trusted: {}",
            failures.join("; ")
        ))
    }
}

/// Every traced kernel name with its request count.
fn trace_counts() -> BTreeMap<String, u64> {
    traced_kernels()
        .into_iter()
        .map(|name| {
            let n = traced_kernel_count(&name);
            (name, n)
        })
        .collect()
}

/// The names whose request count is higher in `after` than in `before`.
fn kernels_that_rose(before: &BTreeMap<String, u64>, after: &BTreeMap<String, u64>) -> BTreeSet<String> {
    after
        .iter()
        .filter(|&(name, &n)| n > before.get(name).copied().unwrap_or(0))
        .map(|(name, _)| name.clone())
        .collect()
}

/// The kernels `work` requested a pipeline for: the names whose trace count
/// rose around it. The device is drained on both sides, so nothing another
/// step encoded is counted here and nothing `work` encoded is left unencoded.
fn kernels_requested_by(rt: &Arc<GpuRuntime>, work: impl FnOnce() -> Res<()>) -> Res<BTreeSet<String>> {
    rt.synchronize()?;
    let before = trace_counts();
    work()?;
    rt.synchronize()?;
    Ok(kernels_that_rose(&before, &trace_counts()))
}

/// Fail unless the real part and the replica group request the same kernels.
/// An empty side fails as "cannot attest" (an op that skips `pipeline` on
/// reuse, or an unarmed trace, would look the same as agreement); a difference
/// names every kernel missing from the replica and every one extra in it.
fn compare_kernel_sets(group: &str, real: &BTreeSet<String>, replica: &BTreeSet<String>) -> Res<()> {
    let empty: Vec<&str> = [("real part", real), ("replica stages", replica)]
        .into_iter()
        .filter(|(_, set)| set.is_empty())
        .map(|(side, _)| side)
        .collect();
    if !empty.is_empty() {
        return Err(format!(
            "{group}: cannot attest: the {} requested no traced kernel",
            empty.join(" and the ")
        ));
    }
    let list = |a: &BTreeSet<String>, b: &BTreeSet<String>| a.difference(b).cloned().collect::<Vec<_>>().join(", ");
    let (missing, extra) = (list(real, replica), list(replica, real));
    if missing.is_empty() && extra.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{group}: the replica requests different kernels than the real part (missing from the replica: [{missing}]; extra in the replica: [{extra}])"
    ))
}

/// Whether `Qwen35Model::prefill_attention` takes the bf16 tile for this
/// precision and head dim (else the f32 kernel); the replica's sdpa mirrors it.
fn uses_bf16_attention(precision: Precision, head_dim: u32) -> bool {
    precision == Precision::Bf16 && head_dim == qwen35::PREFIX_ATTN_HEAD_DIM
}

/// The bf16 Q, K and V of the bf16 attention route, and their element counts.
struct AttnBf16 {
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    n_q: usize,
    n_kv: usize,
}

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
    /// Present exactly when [`uses_bf16_attention`].
    a_bf16: Option<AttnBf16>,
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
            a_bf16: if uses_bf16_attention(model.precision(), l.head_dim()) {
                let bf16b = |n: usize| rt.alloc_buffer(n.max(1) * 2);
                Some(AttnBf16 {
                    q: bf16b(t * qd)?,
                    k: bf16b(t * kvd)?,
                    v: bf16b(t * kvd)?,
                    n_q: t * qd,
                    n_kv: t * kvd,
                })
            } else {
                None
            },
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
        let Some(b) = &r.a_bf16 else {
            return qwen35::attn_prefill(
                self.rt, &r.a_q, &r.a_k, &r.a_v, &r.a_o, &r.tkv, &r.zero, &r.zero, dims, false,
            );
        };
        // The real attention part rounds Q, K and V once, then runs the bf16 tile.
        attn_prefill_bf16::cast_f32_to_bf16(self.rt, &r.a_q, &b.q, b.n_q)?;
        attn_prefill_bf16::cast_f32_to_bf16(self.rt, &r.a_k, &b.k, b.n_kv)?;
        attn_prefill_bf16::cast_f32_to_bf16(self.rt, &r.a_v, &b.v, b.n_kv)?;
        attn_prefill_bf16::attn_prefill_bf16_with_tile(
            self.rt,
            &b.q,
            &b.k,
            &b.v,
            &r.a_o,
            &r.tkv,
            &r.zero,
            &r.zero,
            dims,
            false,
            AttnTile::Q32K64Sg4,
            AttnProbTile::Bf16,
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

    /// The gdn group's stages in forward order (`gdn_core`, not its split).
    fn gdn_pass(&self) -> Res<()> {
        self.norm(&self.r.input_norm)?;
        self.g_in_proj()?;
        self.g_conv()?;
        self.g_scan(None)?;
        self.g_gated_norm()?;
        self.g_out()
    }

    /// The attn group's stages in forward order.
    fn attn_pass(&self) -> Res<()> {
        self.norm(&self.r.input_norm)?;
        self.a_in_proj()?;
        self.a_qk_rope()?;
        self.a_sdpa()?;
        self.a_gate()?;
        self.a_out()
    }

    /// The mlp group's stages in forward order.
    fn mlp_pass(&self) -> Res<()> {
        self.norm(&self.r.input_norm)?;
        self.mlp_gate_up()?;
        self.mlp_swiglu()?;
        self.mlp_down()
    }
}

/// Compare the kernels each replica group requests with those of the real
/// model's parts for the same layer (the first GDN layer for gdn and mlp, the
/// first attention layer for attn, as `time_parts` picks them). Untimed.
/// Every group is compared and every failing group is named in the error.
fn kernel_set_check(rt: &Arc<GpuRuntime>, model: &Qwen35Model, ids: &[u32], ops: &Ops<'_>) -> Res<()> {
    let kinds = &model.config().layers;
    let find = |k: LayerKind| {
        kinds
            .iter()
            .position(|&x| x == k)
            .ok_or_else(|| format!("kernel-set check: the model has no {k:?} layer to compare"))
    };
    let (gdn, attn) = (find(LayerKind::LinearAttention)?, find(LayerKind::FullAttention)?);
    let s = model.begin(ids)?;
    let real = |layer: usize, parts: &[LayerPart]| {
        kernels_requested_by(rt, || parts.iter().try_for_each(|&p| s.encode_part(layer, p)))
    };
    let mixer = [LayerPart::InputNorm, LayerPart::Mixer];
    let mlp = [LayerPart::PostNorm, LayerPart::Mlp];
    let real_sets = [
        ("gdn", real(gdn, &mixer)?),
        ("attn", real(attn, &mixer)?),
        ("mlp", real(gdn, &mlp)?),
    ];
    // The replica in forward order, so every stage reads what the one before wrote.
    ops.embed()?;
    let replica_sets = [
        ("gdn", kernels_requested_by(rt, || ops.gdn_pass())?),
        ("attn", kernels_requested_by(rt, || ops.attn_pass())?),
        ("mlp", kernels_requested_by(rt, || ops.mlp_pass())?),
    ];
    let failures: Vec<String> = real_sets
        .iter()
        .zip(&replica_sets)
        .filter_map(|((group, real), (_, replica))| compare_kernel_sets(group, real, replica).err())
        .collect();
    if failures.is_empty() {
        println!(
            "kernel-set check: compared {} gdn, {} attn, {} mlp kernels",
            real_sets[0].1.len(),
            real_sets[1].1.len(),
            real_sets[2].1.len()
        );
        Ok(())
    } else {
        Err(format!(
            "kernel-set check failed, so the replica's stages are not the real model's work and nothing was timed for this T: {}",
            failures.join("; ")
        ))
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

/// What `run` prints after the per-T lines when some T failed its cross-check:
/// the requested and failed T values, so a log with only the passing T's JSON is
/// never read as a full table (`bench/qwen35_stage_profile_report.py` refuses it).
const INCOMPLETE_PREFIX: &str = "STAGE_PROFILE_INCOMPLETE ";

/// Measure every `t` in `ts`, in order. `measure(t)` returns `Err` for a hard
/// failure (a device or input error: the run stops there), or `Ok(Err(reason))`
/// for a T whose replica did not reproduce the real parts, or `Ok(Ok(json))` for
/// a trusted T. A trusted T's JSON line goes to `emit` as soon as it is known; a
/// failed T is recorded and the loop goes on, so one bad T does not hide whether
/// the others pass. If any T failed, `emit` also gets one `STAGE_PROFILE_INCOMPLETE`
/// line (the requested and failed T values) and the returned error names every
/// failing T with its reason, in the order requested. A hard error at T=x also
/// emits the marker, with `aborted_at` x and `unmeasured` (the requested T values
/// after it), then returns an error naming x, its cause and every earlier failure.
fn profile_every_t(
    ts: &[usize],
    iters: usize,
    mut emit: impl FnMut(&str),
    mut measure: impl FnMut(usize) -> Res<Result<String, String>>,
) -> Res<()> {
    let mut failed: Vec<(usize, String)> = Vec::new();
    let mut aborted: Option<(usize, usize, String)> = None;
    for (i, &t) in ts.iter().enumerate() {
        match measure(t) {
            Ok(Ok(json)) => emit(&json),
            Ok(Err(reason)) => failed.push((t, reason)),
            Err(hard) => {
                aborted = Some((i, t, hard));
                break;
            }
        }
    }
    if failed.is_empty() && aborted.is_none() {
        return Ok(());
    }
    let list = |v: &mut dyn Iterator<Item = usize>| v.map(|t| t.to_string()).collect::<Vec<_>>().join(",");
    let abort_keys = match &aborted {
        Some((i, t, _)) => format!(
            ",\"aborted_at\":{t},\"unmeasured\":[{}]",
            list(&mut ts[i + 1..].iter().copied())
        ),
        None => String::new(),
    };
    emit(&format!(
        "{INCOMPLETE_PREFIX}{{\"requested\":[{}],\"failed\":[{}],\"iters\":{iters}{abort_keys}}}",
        list(&mut ts.iter().copied()),
        list(&mut failed.iter().map(|&(t, _)| t))
    ));
    let named: Vec<String> = failed.iter().map(|(t, r)| format!("T={t}: {r}")).collect();
    match aborted {
        None => Err(format!(
            "{} of {} requested T values failed the replica cross-check (JSON emitted only for the others): {}",
            failed.len(),
            ts.len(),
            named.join(" | ")
        )),
        Some((i, t, hard)) => Err(format!(
            "run aborted at T={t}: {hard}; {} requested T values after it were not measured; {} earlier T values failed the replica cross-check{}{}",
            ts.len() - i - 1,
            failed.len(),
            if named.is_empty() { "" } else { ": " },
            named.join(" | ")
        )),
    }
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
    let trace_on = kernel_trace_enabled();
    if !trace_on {
        println!("kernel-set check: NOT RUN (TESSL_KERNEL_TRACE unset)");
    }
    // Restore the flag the caller had on every path, including a failed T.
    let prior_async = rt.async_encode_enabled();
    rt.set_async_encode(true)?;
    let profiled = profile_every_t(
        ts,
        iters,
        |line| println!("{line}"),
        |t| {
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
            // Structural drift is a hard error before any timing.
            if trace_on {
                kernel_set_check(rt, model, ids, &ops)?;
                println!("kernel-set check: PASS T={t}");
            }
            // One pass in forward order so every stage's input is what the stage
            // before it wrote; then each stage alone.
            ops.embed()?;
            ops.gdn_pass()?;
            ops.attn_pass()?;
            ops.mlp_pass()?;
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
            let verdict = check_replica_against_parts(
                &[
                    ("gdn layer", parts.gdn_layer, gdn_stage),
                    ("attn layer", parts.attn_layer, attn_stage),
                    ("mlp", parts.mlp, mlp_stage),
                ],
                REPLICA_REL_TOL,
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
            let line = format!(
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
            Ok(verdict.map(|()| line))
        },
    );
    let restored = rt.set_async_encode(prior_async);
    match (profiled, restored) {
        (Ok(()), restored) => restored,
        (Err(e), Ok(())) => Err(e),
        (Err(e), Err(r)) => Err(format!("{e}; restoring async encode also failed: {r}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_within_tolerance_pass() {
        // The committed 1k evidence: GDN is 3.19% off, the worst row.
        let rows = [
            ("gdn", 2.9409, 3.0348),
            ("attn", 2.0793, 2.104),
            ("mlp", 2.9416, 2.9735),
        ];
        assert!(check_replica_against_parts(&rows, REPLICA_REL_TOL).is_ok());
    }

    #[test]
    fn the_tolerance_is_five_percent() {
        assert_eq!(REPLICA_REL_TOL, 0.05);
    }

    #[test]
    fn a_row_just_over_tolerance_fails_and_is_named() {
        let rows = [("gdn", 100.0, 105.1), ("attn", 100.0, 100.0), ("mlp", 100.0, 100.0)];
        let err = check_replica_against_parts(&rows, REPLICA_REL_TOL).unwrap_err();
        assert!(err.contains("gdn"), "{err}");
        assert!(!err.contains("attn") && !err.contains("mlp"), "{err}");
        assert!(err.contains("100") && err.contains("105.1"), "{err}");
    }

    #[test]
    fn exactly_at_the_boundary_passes_in_both_directions() {
        let rows = [("hi", 100.0, 105.0), ("lo", 100.0, 95.0)];
        assert!(check_replica_against_parts(&rows, REPLICA_REL_TOL).is_ok());
    }

    #[test]
    fn a_replica_that_is_too_fast_fails_too() {
        let rows = [("mlp", 100.0, 94.0)];
        assert!(check_replica_against_parts(&rows, REPLICA_REL_TOL).is_err());
    }

    #[test]
    fn uncomparable_rows_fail_closed() {
        for (real, replica) in [
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (-2.0, 2.0),
        ] {
            let err = check_replica_against_parts(&[("row", real, replica)], REPLICA_REL_TOL)
                .expect_err(&format!("real {real} replica {replica} must not pass"));
            assert!(err.contains("row"), "{err}");
        }
    }

    #[test]
    fn an_empty_table_is_not_a_pass() {
        assert!(check_replica_against_parts(&[], REPLICA_REL_TOL).is_err());
    }

    #[test]
    fn every_failing_row_is_named() {
        let rows = [("gdn", 10.0, 12.0), ("attn", 10.0, 10.1), ("mlp", 10.0, f64::NAN)];
        let err = check_replica_against_parts(&rows, REPLICA_REL_TOL).unwrap_err();
        assert!(err.contains("gdn") && err.contains("mlp"), "{err}");
        assert!(!err.contains("attn"), "{err}");
    }

    #[test]
    fn a_zero_or_negative_replica_against_a_positive_real_fails() {
        for replica in [0.0, -0.0, -1.0, -100.0, -(f64::MIN_POSITIVE / 2.0)] {
            let err = check_replica_against_parts(&[("row", 100.0, replica)], REPLICA_REL_TOL)
                .expect_err(&format!("replica {replica} must not pass"));
            assert!(err.contains("row"), "{err}");
        }
    }

    #[test]
    fn negative_zero_real_is_not_comparable() {
        let err = check_replica_against_parts(&[("row", -0.0, 0.0)], REPLICA_REL_TOL).unwrap_err();
        assert!(err.contains("not comparable"), "{err}");
    }

    #[test]
    fn huge_times_do_not_overflow_into_a_pass() {
        let big = f64::MAX;
        // The gap |replica - real| is infinity here; it must still fail.
        assert!(check_replica_against_parts(&[("row", big, -big)], REPLICA_REL_TOL).is_err());
        assert!(check_replica_against_parts(&[("row", big, 0.0)], REPLICA_REL_TOL).is_err());
        assert!(check_replica_against_parts(&[("row", big, big * 0.9)], REPLICA_REL_TOL).is_err());
        assert!(check_replica_against_parts(&[("row", big, big)], REPLICA_REL_TOL).is_ok());
        assert!(check_replica_against_parts(&[("row", big, big * 0.97)], REPLICA_REL_TOL).is_ok());
        // tol * real stays finite for every finite real, and the replica may not exceed it by 5%.
        assert!(check_replica_against_parts(&[("row", big * 0.5, big)], REPLICA_REL_TOL).is_err());
    }

    #[test]
    fn subnormal_times_compare_exactly() {
        let tiny = 5e-324_f64;
        assert!(tiny > 0.0 && tiny < f64::MIN_POSITIVE);
        // tol * tiny underflows to 0, so only an identical replica passes.
        assert!(check_replica_against_parts(&[("row", tiny, tiny)], REPLICA_REL_TOL).is_ok());
        assert!(check_replica_against_parts(&[("row", tiny, 2.0 * tiny)], REPLICA_REL_TOL).is_err());
        assert!(check_replica_against_parts(&[("row", tiny, 0.0)], REPLICA_REL_TOL).is_err());
    }

    #[test]
    fn floating_point_boundary_rows_follow_the_computed_comparison() {
        // 3.15 - 3.0 = 0.1499999999999999 <= 0.05 * 3.0 = 0.15000000000000002: inclusive, passes.
        assert!(check_replica_against_parts(&[("a", 3.0, 3.15)], REPLICA_REL_TOL).is_ok());
        assert!(check_replica_against_parts(&[("b", 7.0, 7.35)], REPLICA_REL_TOL).is_ok());
        assert!(check_replica_against_parts(&[("c", 20.0, 21.0)], REPLICA_REL_TOL).is_ok());
        // The literal 1.05 is a hair over 1.05 in binary (gap 0.050000000000000044 > 0.05): fails.
        assert!(check_replica_against_parts(&[("d", 1.0, 1.05)], REPLICA_REL_TOL).is_err());
    }

    #[test]
    fn a_zero_tolerance_demands_identical_times() {
        assert!(check_replica_against_parts(&[("row", 2.5, 2.5)], 0.0).is_ok());
        assert!(check_replica_against_parts(&[("row", 2.5, 2.5000000000000004)], 0.0).is_err());
    }

    #[test]
    fn a_negative_tolerance_fails_closed_even_for_equal_rows() {
        assert!(check_replica_against_parts(&[("row", 2.5, 2.5)], -0.05).is_err());
    }

    #[test]
    fn defect_a_nan_tolerance_must_not_pass_any_row() {
        // `gap > NaN` is false, so a NaN tolerance accepts a replica that is 10x off.
        // Only the const reaches this fn today; the fn does not validate rel_tol itself.
        let r = check_replica_against_parts(&[("row", 1.0, 10.0)], f64::NAN);
        assert!(r.is_err(), "NaN tolerance passed a 900% gap");
    }

    #[test]
    fn defect_an_infinite_tolerance_must_not_pass_any_row() {
        let r = check_replica_against_parts(&[("row", 1.0, 10.0)], f64::INFINITY);
        assert!(r.is_err(), "infinite tolerance passed a 900% gap");
    }

    #[test]
    fn the_bf16_attention_route_needs_bf16_and_the_256_wide_head() {
        assert!(uses_bf16_attention(Precision::Bf16, 256));
        assert!(!uses_bf16_attention(Precision::F32, 256));
        assert!(!uses_bf16_attention(Precision::Bf16, 128));
        assert!(!uses_bf16_attention(Precision::Bf16, 0));
    }

    fn set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn equal_kernel_sets_pass() {
        let s = set(&["rms_norm_bf16", "cast_f32_to_bf16"]);
        assert!(compare_kernel_sets("attn", &s, &s.clone()).is_ok());
    }

    #[test]
    fn a_kernel_missing_from_the_replica_fails_and_is_named() {
        let err = compare_kernel_sets("attn", &set(&["a", "cast_f32_to_bf16"]), &set(&["a"])).unwrap_err();
        assert!(err.starts_with("attn:"), "{err}");
        assert!(err.contains("missing from the replica: [cast_f32_to_bf16]"), "{err}");
        assert!(err.contains("extra in the replica: []"), "{err}");
    }

    #[test]
    fn a_kernel_extra_in_the_replica_fails_and_is_named() {
        let err = compare_kernel_sets("mlp", &set(&["a"]), &set(&["a", "qwen35_attn_f32"])).unwrap_err();
        assert!(err.contains("extra in the replica: [qwen35_attn_f32]"), "{err}");
        assert!(err.contains("missing from the replica: []"), "{err}");
    }

    #[test]
    fn a_swapped_kernel_names_both_sides() {
        let err = compare_kernel_sets("attn", &set(&["tile_bf16", "x"]), &set(&["tile_f32", "x"])).unwrap_err();
        assert!(err.contains("missing from the replica: [tile_bf16]"), "{err}");
        assert!(err.contains("extra in the replica: [tile_f32]"), "{err}");
    }

    #[test]
    fn an_empty_side_cannot_attest_and_is_never_a_pass() {
        let some = set(&["a"]);
        let none = BTreeSet::new();
        for (real, replica, who) in [(&none, &some, "real part"), (&some, &none, "replica stages")] {
            let err = compare_kernel_sets("gdn", real, replica).unwrap_err();
            assert!(err.contains("cannot attest") && err.contains(who), "{err}");
        }
        // Two empty sets are equal but attest nothing: still an error.
        let err = compare_kernel_sets("gdn", &none, &none.clone()).unwrap_err();
        assert!(
            err.contains("cannot attest") && err.contains("real part") && err.contains("replica stages"),
            "{err}"
        );
    }

    #[test]
    fn only_names_whose_count_rose_are_in_the_delta() {
        let before: BTreeMap<String, u64> = [("a", 3), ("b", 1)].map(|(n, c)| (n.to_string(), c)).into();
        let after: BTreeMap<String, u64> = [("a", 3), ("b", 2), ("c", 1)].map(|(n, c)| (n.to_string(), c)).into();
        assert_eq!(kernels_that_rose(&before, &after), set(&["b", "c"]));
        // A cache-hit request raises a count without adding a name; nothing requested adds nothing.
        assert!(kernels_that_rose(&after, &after.clone()).is_empty());
        assert!(kernels_that_rose(&BTreeMap::new(), &BTreeMap::new()).is_empty());
    }

    #[test]
    fn a_kernel_set_mismatch_aborts_the_run_before_later_t_values() {
        let mut emitted = Vec::new();
        let err = profile_every_t(
            &[1024, 2048],
            3,
            |l| emitted.push(l.to_string()),
            |t| {
                if t == 1024 {
                    compare_kernel_sets("attn", &set(&["a", "b"]), &set(&["a"]))?;
                }
                Ok(Ok(format!("json {t}")))
            },
        )
        .unwrap_err();
        assert!(
            err.contains("run aborted at T=1024") && err.contains("missing from the replica: [b]"),
            "{err}"
        );
        assert!(!emitted.iter().any(|l| l.starts_with("json")), "{emitted:?}");
        assert!(
            emitted
                .iter()
                .any(|l| l.contains("\"aborted_at\":1024") && l.contains("\"unmeasured\":[2048]")),
            "{emitted:?}"
        );
    }

    /// Drives `profile_every_t` with a fake measurer; returns (measured T's, emitted lines, result).
    fn drive(ts: &[usize], fail: &[usize]) -> (Vec<usize>, Vec<String>, Res<()>) {
        let (mut measured, mut emitted) = (Vec::new(), Vec::new());
        let r = profile_every_t(
            ts,
            3,
            |l| emitted.push(l.to_string()),
            |t| {
                measured.push(t);
                Ok(if fail.contains(&t) {
                    Err(format!("gap at {t}"))
                } else {
                    Ok(format!("json {t}"))
                })
            },
        );
        (measured, emitted, r)
    }

    #[test]
    fn a_failing_middle_t_does_not_hide_the_later_one() {
        let (measured, emitted, r) = drive(&[1024, 2048, 8192], &[2048]);
        assert_eq!(measured, [1024, 2048, 8192], "every T must be measured");
        assert_eq!(emitted[..2], ["json 1024", "json 8192"]);
        assert_eq!(
            emitted.len(),
            3,
            "two JSON lines and one incomplete marker: {emitted:?}"
        );
        assert_eq!(
            emitted[2],
            "STAGE_PROFILE_INCOMPLETE {\"requested\":[1024,2048,8192],\"failed\":[2048],\"iters\":3}"
        );
        let err = r.unwrap_err();
        assert!(err.contains("T=2048: gap at 2048"), "{err}");
        assert!(!err.contains("T=1024") && !err.contains("T=8192"), "{err}");
    }

    #[test]
    fn every_failing_t_is_named_in_order_and_no_json_is_emitted() {
        let (measured, emitted, r) = drive(&[1024, 8192], &[1024, 8192]);
        assert_eq!(measured, [1024, 8192]);
        assert_eq!(emitted.len(), 1, "only the incomplete marker: {emitted:?}");
        assert!(emitted[0].starts_with(INCOMPLETE_PREFIX), "{emitted:?}");
        let err = r.unwrap_err();
        let (a, b) = (err.find("T=1024: gap at 1024"), err.find("T=8192: gap at 8192"));
        assert!(a.is_some() && b.is_some() && a < b, "{err}");
        assert!(err.contains("2 of 2"), "{err}");
    }

    #[test]
    fn all_passing_ts_emit_each_json_and_no_marker() {
        let (measured, emitted, r) = drive(&[1024, 8192], &[]);
        assert_eq!(measured, [1024, 8192]);
        assert_eq!(emitted, ["json 1024", "json 8192"]);
        assert!(r.is_ok());
    }

    #[test]
    fn an_empty_t_list_is_a_silent_ok() {
        let (measured, emitted, r) = drive(&[], &[]);
        assert!(measured.is_empty() && emitted.is_empty() && r.is_ok());
    }

    #[test]
    fn a_hard_error_stops_the_run_with_a_marker_and_no_later_ts() {
        let (mut measured, mut emitted) = (Vec::new(), Vec::new());
        let r = profile_every_t(
            &[1, 2, 3],
            3,
            |l| emitted.push(l.to_string()),
            |t| {
                measured.push(t);
                if t == 2 {
                    Err("device lost".to_string())
                } else {
                    Ok(Ok(format!("json {t}")))
                }
            },
        );
        assert_eq!(measured, [1, 2]);
        assert_eq!(emitted.len(), 2, "{emitted:?}");
        assert_eq!(emitted[0], "json 1");
        assert_eq!(
            emitted[1],
            "STAGE_PROFILE_INCOMPLETE {\"requested\":[1,2,3],\"failed\":[],\"iters\":3,\"aborted_at\":2,\"unmeasured\":[3]}"
        );
        let err = r.unwrap_err();
        assert!(err.contains("T=2: device lost") && err.contains("1 requested"), "{err}");
    }

    #[test]
    fn a_hard_error_after_a_soft_failure_names_both_and_marks_both() {
        let mut emitted = Vec::new();
        let r = profile_every_t(
            &[1, 2, 3],
            3,
            |l| emitted.push(l.to_string()),
            |t| match t {
                1 => Ok(Err("gap at 1".into())),
                2 => Err("device lost".into()),
                _ => Ok(Ok("json 3".into())),
            },
        );
        assert_eq!(
            emitted,
            ["STAGE_PROFILE_INCOMPLETE {\"requested\":[1,2,3],\"failed\":[1],\"iters\":3,\"aborted_at\":2,\"unmeasured\":[3]}"]
        );
        let err = r.unwrap_err();
        assert!(
            err.contains("T=2: device lost") && err.contains("T=1: gap at 1"),
            "{err}"
        );
    }

    #[test]
    fn a_hard_error_at_the_last_t_has_an_empty_unmeasured_list() {
        let mut emitted = Vec::new();
        let r = profile_every_t(
            &[4, 5],
            1,
            |l| emitted.push(l.to_string()),
            |t| {
                if t == 5 {
                    Err("x".into())
                } else {
                    Ok(Ok("json 4".into()))
                }
            },
        );
        assert!(r.is_err());
        assert!(
            emitted[1].ends_with("\"aborted_at\":5,\"unmeasured\":[]}"),
            "{emitted:?}"
        );
    }

    /// Strict reader for the marker payload `{"requested":[..],"failed":[..],"iters":N}`:
    /// ASCII digits only, exact key order, no whitespace. Anything else is `None`.
    fn parse_marker(line: &str) -> Option<(Vec<usize>, Vec<usize>, usize)> {
        let body = line.strip_prefix(INCOMPLETE_PREFIX)?;
        let body = body.strip_prefix("{\"requested\":[")?;
        let (req, rest) = body.split_once("],\"failed\":[")?;
        let (fail, rest) = rest.split_once("],\"iters\":")?;
        let iters = rest.strip_suffix('}')?;
        let nums = |s: &str| -> Option<Vec<usize>> {
            if s.is_empty() {
                return Some(Vec::new());
            }
            s.split(',')
                .map(|n| {
                    (!n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                        .then(|| n.parse().ok())
                        .flatten()
                })
                .collect()
        };
        if iters.is_empty() || !iters.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some((nums(req)?, nums(fail)?, iters.parse().ok()?))
    }

    #[test]
    fn attack_duplicate_unsorted_and_zero_ts_keep_request_order() {
        let (measured, emitted, r) = drive(&[8192, 0, 1024, 1024, 0], &[0, 1024]);
        assert_eq!(measured, [8192, 0, 1024, 1024, 0]);
        assert_eq!(emitted[0], "json 8192");
        let (req, failed, iters) = parse_marker(emitted.last().unwrap()).expect("marker parses");
        assert_eq!(req, [8192, 0, 1024, 1024, 0]);
        assert_eq!(failed, [0, 1024, 1024, 0]);
        assert_eq!(iters, 3);
        assert!(r.unwrap_err().contains("4 of 5"));
    }

    #[test]
    fn attack_a_duplicate_t_that_passes_once_and_fails_once_still_marks_incomplete() {
        let mut calls = 0;
        let mut emitted = Vec::new();
        let r = profile_every_t(
            &[7, 7],
            1,
            |l| emitted.push(l.to_string()),
            |_| {
                calls += 1;
                Ok(if calls == 1 {
                    Ok("json 7".into())
                } else {
                    Err("flaky".into())
                })
            },
        );
        assert!(r.is_err());
        assert_eq!(emitted[0], "json 7");
        assert_eq!(parse_marker(&emitted[1]).unwrap().1, [7]);
    }

    #[test]
    fn attack_hostile_reasons_cannot_reach_the_marker() {
        let hostile = [
            "",
            "a | T=9: forged",
            "line1\nSTAGE_PROFILE_INCOMPLETE {\"requested\":[],\"failed\":[],\"iters\":0}",
            "quote\" and \\ and \r\n",
            "STAGE_PROFILE_JSON {}",
        ];
        for reason in hostile {
            let mut emitted = Vec::new();
            let r = profile_every_t(
                &[5, 6],
                2,
                |l| emitted.push(l.to_string()),
                |t| {
                    Ok(if t == 5 {
                        Err(reason.to_string())
                    } else {
                        Ok("json 6".into())
                    })
                },
            );
            assert_eq!(emitted.len(), 2, "{reason:?}: {emitted:?}");
            assert!(emitted.iter().all(|l| !l.contains('\n') && !l.contains('\r')));
            assert_eq!(parse_marker(&emitted[1]), Some((vec![5, 6], vec![5], 2)), "{reason:?}");
            let err = r.unwrap_err();
            assert!(err.contains("T=5: "), "{err}");
            assert!(err.contains("1 of 2"), "{err}");
        }
    }

    #[test]
    fn attack_a_huge_t_list_keeps_a_valid_marker() {
        let ts: Vec<usize> = (0..200_000).collect();
        let fail: Vec<usize> = ts.iter().copied().filter(|t| t % 3 == 0).collect();
        let (measured, emitted, r) = drive(&ts, &fail);
        assert_eq!(measured.len(), ts.len());
        let (req, failed, _) = parse_marker(emitted.last().unwrap()).expect("marker parses");
        assert_eq!((req.len(), failed), (ts.len(), fail));
        assert!(r.unwrap_err().contains(&format!("{} of 200000", 66_667)));
    }

    #[test]
    fn attack_the_marker_is_emitted_after_every_json_line() {
        let (_, emitted, _) = drive(&[1, 2, 3, 4], &[1, 3]);
        assert_eq!(emitted.len(), 3);
        assert!(emitted[2].starts_with(INCOMPLETE_PREFIX));
        assert!(emitted[..2].iter().all(|l| !l.starts_with(INCOMPLETE_PREFIX)));
    }

    /// A hard error after at least one T was emitted leaves a stdout with passing
    /// JSON and no marker: `bench/qwen35_stage_profile_report.py` accepts that as a
    /// full table, though the run is partial. The old expectation
    /// (`a_hard_error_stops_the_run_without_a_marker_or_later_ts`) encoded this defect.
    #[test]
    fn defect_a_hard_error_after_a_pass_leaves_a_partial_log_without_a_marker() {
        let mut emitted = Vec::new();
        let r = profile_every_t(
            &[1, 2, 3],
            3,
            |l| emitted.push(l.to_string()),
            |t| {
                if t == 2 {
                    Err("device lost".into())
                } else {
                    Ok(Ok(format!("json {t}")))
                }
            },
        );
        assert!(r.is_err());
        assert_eq!(emitted[0], "json 1");
        assert!(
            emitted.iter().any(|l| l.starts_with(INCOMPLETE_PREFIX)),
            "stdout holds a passing T and no INCOMPLETE marker, so a log-only consumer reads it as complete: {emitted:?}"
        );
    }

    /// A hard error after a cross-check failure drops the failed T from both the
    /// marker and the error text.
    #[test]
    fn defect_a_hard_error_after_a_soft_failure_forgets_the_failed_t() {
        let mut emitted = Vec::new();
        let r = profile_every_t(
            &[1, 2, 3],
            3,
            |l| emitted.push(l.to_string()),
            |t| match t {
                1 => Ok(Err("gap at 1".into())),
                2 => Err("device lost".into()),
                _ => Ok(Ok("json 3".into())),
            },
        );
        let err = r.unwrap_err();
        assert!(err.contains("device lost"), "{err}");
        assert!(
            err.contains("T=1"),
            "the earlier cross-check failure is not named: {err}"
        );
        assert!(
            emitted.iter().any(|l| l.starts_with(INCOMPLETE_PREFIX)),
            "no marker: {emitted:?}"
        );
    }
}
