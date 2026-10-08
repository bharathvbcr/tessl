//! Training-path throughput at Qwen3.5-2B's shapes: the training forward
//! ops, every backward op, and optionally one whole training step.
//!
//! ```text
//! cargo run --release --bin bench_qwen35_train                  # T = 2048
//! cargo run --release --bin bench_qwen35_train -- 1024 4096     # chosen T
//! cargo run --release --bin bench_qwen35_train -- --check-only  # the gate, no timing
//! QWEN35_2B_SAFETENSORS=... cargo run --release --bin bench_qwen35_train -- --step=2048
//! cargo run --release --bin bench_qwen35_train -- --bf16        # bf16 GEMM operands
//! QWEN35_2B_SAFETENSORS=... cargo run --release --bin bench_qwen35_train -- 2048 --step-only --step=2048 --async
//! QWEN35_2B_SAFETENSORS=... cargo run --release --bin bench_qwen35_train -- 256 --step-only --clip=256
//! ```
//!
//! Shapes are Qwen3.5-2B's `text_config` (hidden 2048, 16 GDN heads of 128,
//! 8 query and 2 KV attention heads of 256, rotary 64, MLP 6144, vocab
//! 248320), batch 1, f32 operands. The GEMMs (the cross-entropy's here, every
//! one in `--step`) run on exact f32 operands, or on bf16 operands with f32
//! accumulation under `--bf16`. Inputs are
//! random and bounded.
//!
//! Per op, the time of one call at the given T: the median of 7 runs after 2
//! warm-ups, each run `REPS` calls encoded into one command buffer and
//! divided, except the cross-entropy and the embedding backward, which write
//! host-side row lists per call and are timed one call per run (so they carry
//! the ~0.25 ms submit-and-wait floor, `docs/benchmarking.md`). A GDN layer's
//! training ops are `gdn_train` forward + backward; an attention layer's are
//! `attn_train` forward + backward; the row-local backwards are listed apart.
//!
//! `--batch=ROWS,LEN[,SPAN_ROWS]` (repeatable, with the real checkpoint)
//! times one optimizer step's gradients for a batch run row by row into one
//! bank, as a padded batch runs through tessl: letter rows supervise one
//! position, span rows hand hidden rows to an outside loss and take its
//! gradient back (span rows need LEN >= 5). Two lengths at one row count
//! separate the per-row fixed cost from the per-token cost. The warm-up's
//! gradient norm must be finite and non-zero before it is timed.
//!
//! `--step=N` also loads the real checkpoint (`QWEN35_2B_SAFETENSORS`) and
//! times `Qwen35Model::train_step` on N tokens end to end (median of 3 after
//! one warm-up), which includes allocating every rebuilt activation and
//! gradient per step. Run it under `/usr/bin/time -l` for its peak memory.
//! It also prints the device's peak allocation over the steps
//! ([`GpuRuntime::peak_allocated_bytes`], reset after loading), and one more
//! untimed step's host waits on the GPU, commits, dispatches and fresh
//! buffer allocations ([`tessl::infer_trace`]).
//!
//! `--async` runs the step under [`GpuRuntime::set_async_encode`] (one command
//! buffer between the waits the step itself makes); without it every
//! dispatch is a waited commit, as on a runtime nobody switched (ojas-qwen35's
//! `Qwen35Step`). `--step-only` runs the per-op gate without timing the ops.
//!
//! `--clip=N` times one clipped optimizer step on N tokens as a trainer runs
//! it: `train_step_into` a bank, `grad_sq_norm` of the bank, then
//! `Qwen35Model::adamw_step` with the clip coefficient, each phase's wall time
//! and host waits counted (median of 3 after one warm-up). It holds the bank
//! and f32 AdamW moments beside the model (4 parameter copies, ~30 GB on the
//! 2B).
//!
//! Before anything is timed, every op runs once with its outputs pre-filled
//! with NaN, and every output must come back finite: an op that silently
//! wrote nothing would otherwise post the best number. There is no timing
//! without the gate.

use std::sync::Arc;
use std::time::Instant;

use tessl::attn_train::{attn_train_backward, attn_train_forward, AttnTrainDims, AttnTrainGrads, AttnTrainWorkspace};
use tessl::cross_entropy::{cross_entropy_rows, CeGrads, CeHidden, CeWorkspace, Reduction};
use tessl::gdn_train::{
    gdn_train_backward, gdn_train_forward, GdnTrainDims, GdnTrainGrads, GdnTrainInputs, GdnTrainWorkspace,
};
use tessl::gemm::GemmOperands;
use tessl::qwen35::{AttnShape, Cols, GdnGateLogits, GdnParams};
use tessl::qwen35_adamw::{AdamW, AdamWHyper};
use tessl::qwen35_bwd::{
    attn_qk_norm_rope_bwd, attn_qk_norm_rope_bwd_part_len, conv1d_silu_bwd, conv1d_silu_bwd_part_len, embed_rows_bwd,
    gated_rms_norm_bwd, gated_rms_norm_bwd_part_len, gdn_gates_bwd, gdn_gates_bwd_part_len, rms_norm_bwd,
    rms_norm_bwd_part_len, swiglu_bwd, AttnQkvGrads, EmbedBwdWorkspace,
};
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::Qwen35Grads;
use tessl::safetensors::SafeTensors;
use tessl::tensor::GpuBuffer;
use tessl::{DType, GpuRuntime, Tensor};

const HIDDEN: usize = 2048;
const INTER: usize = 6144;
const VOCAB: usize = 248_320;
const GDN_HEADS: usize = 16;
const GDN_DK: usize = 128;
const GDN_DV: usize = 128;
const CONV_DIM: usize = 2 * GDN_HEADS * GDN_DK + GDN_HEADS * GDN_DV;
const CONV_KW: u32 = 4;
const Q_HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const ROTARY: u32 = 64;
const THETA: f32 = 1e7;
const EPS: f32 = 1e-6;
const CE_CHUNK: u32 = 8192;

const REPS: usize = 4;
const WARMUP: usize = 2;
const ITERS: usize = 7;

type Res<T> = Result<T, String>;

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

fn tensor(rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64, scale: f32) -> Res<Tensor> {
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.write_f32(&fill(shape.iter().product(), seed, scale));
    Ok(t)
}

fn buf(rt: &Arc<GpuRuntime>, n: usize, seed: u64, scale: f32) -> Res<GpuBuffer> {
    let b = rt.alloc_buffer(n.max(1) * 4)?;
    b.write_f32(&fill(n, seed, scale));
    Ok(b)
}

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

/// One timed op: its name, how to run it, the buffers it writes (each with
/// its element count) for the gate, and the calls per timed run.
struct Op<'a> {
    name: &'static str,
    run: Box<dyn FnMut() -> Res<()> + 'a>,
    outputs: Vec<(&'static str, &'a GpuBuffer, usize)>,
    reps: usize,
}

fn gate(rt: &Arc<GpuRuntime>, op: &mut Op<'_>) -> Res<()> {
    for (_, b, n) in &op.outputs {
        let mut all = b.read_f32();
        all[..*n].fill(f32::NAN);
        b.write_f32(&all);
    }
    (op.run)()?;
    rt.synchronize()?;
    for (name, b, n) in &op.outputs {
        let v = b.read_f32();
        if let Some(i) = v[..*n].iter().position(|x| !x.is_finite()) {
            return Err(format!(
                "{}: {name}[{i}] of {n} is {} (unwritten or overflowed)",
                op.name, v[i]
            ));
        }
    }
    Ok(())
}

fn bench_t(rt: &Arc<GpuRuntime>, t: usize, check_only: bool, operands: GemmOperands) -> Res<()> {
    let t32 = t as u32;
    let (qd, kvd) = (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let s = |i: u64| (t as u64) * 1000 + i;

    // GDN: q, k, v, g, beta as gdn_train takes them, and their gradients.
    let gd = GdnTrainDims {
        batch: 1,
        seq: t32,
        heads: GDN_HEADS as u32,
        v_dim: GDN_DV as u32,
    };
    let (q, k, v) = (
        tensor(rt, &[1, t, GDN_HEADS, GDN_DK], s(1), 1.0)?,
        tensor(rt, &[1, t, GDN_HEADS, GDN_DK], s(2), 1.0)?,
        tensor(rt, &[1, t, GDN_HEADS, GDN_DV], s(3), 1.0)?,
    );
    let g = rt.alloc_tensor_f32(&[1, t, GDN_HEADS])?;
    g.buffer.write_f32(
        &fill(t * GDN_HEADS, s(4), 1.0)
            .iter()
            .map(|x| -0.5 - x.abs())
            .collect::<Vec<_>>(),
    );
    let beta = rt.alloc_tensor_f32(&[1, t, GDN_HEADS])?;
    beta.buffer.write_f32(
        &fill(t * GDN_HEADS, s(5), 1.0)
            .iter()
            .map(|x| 0.5 + 0.4 * x)
            .collect::<Vec<_>>(),
    );
    let (o, ckpt) = (
        rt.alloc_tensor_f32(&[1, t, GDN_HEADS, GDN_DV])?,
        rt.alloc_tensor_f32(&gd.checkpoint_shape())?,
    );
    let d_o = tensor(rt, &[1, t, GDN_HEADS, GDN_DV], s(6), 1.0)?;
    let gws = GdnTrainWorkspace::new(rt, gd)?;
    let (dq, dk, dv) = (
        rt.alloc_tensor_f32(&[1, t, GDN_HEADS, GDN_DK])?,
        rt.alloc_tensor_f32(&[1, t, GDN_HEADS, GDN_DK])?,
        rt.alloc_tensor_f32(&[1, t, GDN_HEADS, GDN_DV])?,
    );
    let (dg, dbeta) = (
        rt.alloc_tensor_f32(&[1, t, GDN_HEADS])?,
        rt.alloc_tensor_f32(&[1, t, GDN_HEADS])?,
    );
    let inputs = || GdnTrainInputs {
        q: &q,
        k: &k,
        v: &v,
        g: &g,
        beta: &beta,
        s0: None,
    };

    // Attention: q, k, v, o, lse and their gradients.
    let ad = AttnTrainDims {
        batch: 1,
        seq: t32,
        q_heads: Q_HEADS as u32,
        kv_heads: KV_HEADS as u32,
        scale: 1.0 / 16.0,
    };
    let (aq, ak, av) = (
        buf(rt, t * qd, s(10), 2.0)?,
        buf(rt, t * kvd, s(11), 2.0)?,
        buf(rt, t * kvd, s(12), 1.0)?,
    );
    let (ao, lse, ado) = (
        buf(rt, t * qd, 0, 0.0)?,
        buf(rt, ad.lse_len(), 0, 0.0)?,
        buf(rt, t * qd, s(13), 1.0)?,
    );
    let (adq, adk, adv) = (
        buf(rt, t * qd, 0, 0.0)?,
        buf(rt, t * kvd, 0, 0.0)?,
        buf(rt, t * kvd, 0, 0.0)?,
    );
    let aws = AttnTrainWorkspace::new(rt, ad)?;
    attn_train_forward(rt, &ad, &aq, &ak, &av, &ao, &lse, &aws)?;

    // The cross-entropy over the T - 1 predicted rows.
    let h = tensor(rt, &[t, HIDDEN], s(20), 1.0)?;
    let emb = rt.alloc_tensor_bf16(&[VOCAB, HIDDEN])?;
    emb.buffer.write_bf16_bits(
        &fill(VOCAB * HIDDEN, s(21), 0.05)
            .iter()
            .map(|x| (x.to_bits() >> 16) as u16)
            .collect::<Vec<_>>(),
    );
    let rows: Vec<u32> = (0..t32 - 1).collect();
    let targets: Vec<u32> = (0..t32 - 1).map(|i| (i * 7919) % VOCAB as u32).collect();
    let ce_ws = CeWorkspace::new(rt, t32 - 1, HIDDEN as u32, CE_CHUNK, DType::BF16)?;
    let (dh, dw) = (
        rt.alloc_tensor_f32(&[t - 1, HIDDEN])?,
        rt.alloc_tensor_f32(&[VOCAB, HIDDEN])?,
    );

    // Row-local backward operands.
    let width = CONV_DIM + GDN_HEADS * GDN_DV + 2 * GDN_HEADS; // the GDN fused projection
    let proj = buf(rt, t * width, s(30), 1.0)?;
    let dproj = buf(rt, t * width, 0, 0.0)?;
    let (x, dy_h, dx_h) = (
        buf(rt, t * HIDDEN, s(31), 1.0)?,
        buf(rt, t * HIDDEN, s(32), 1.0)?,
        buf(rt, t * HIDDEN, 0, 0.0)?,
    );
    let (nw, dnw) = (buf(rt, HIDDEN, s(33), 1.0)?, buf(rt, HIDDEN, 0, 0.0)?);
    let npart = buf(rt, rms_norm_bwd_part_len(t32, HIDDEN as u32), 0, 0.0)?;
    let vd = GDN_HEADS * GDN_DV;
    let (go, gdy, gdo) = (
        buf(rt, t * vd, s(34), 1.0)?,
        buf(rt, t * vd, s(35), 1.0)?,
        buf(rt, t * vd, 0, 0.0)?,
    );
    let (gnw, gdnw) = (buf(rt, GDN_DV, s(36), 1.0)?, buf(rt, GDN_DV, 0, 0.0)?);
    let gpart = buf(
        rt,
        gated_rms_norm_bwd_part_len(t32, GDN_HEADS as u32, GDN_DV as u32),
        0,
        0.0,
    )?;
    let (mg, mu, mdy) = (
        buf(rt, t * INTER, s(37), 1.0)?,
        buf(rt, t * INTER, s(38), 1.0)?,
        buf(rt, t * INTER, s(39), 1.0)?,
    );
    let (mdg, mdu) = (buf(rt, t * INTER, 0, 0.0)?, buf(rt, t * INTER, 0, 0.0)?);
    let (cw, cdy, cdw) = (
        buf(rt, CONV_DIM * CONV_KW as usize, s(40), 0.5)?,
        buf(rt, t * CONV_DIM, s(41), 1.0)?,
        buf(rt, CONV_DIM * CONV_KW as usize, 0, 0.0)?,
    );
    let cpart = buf(rt, conv1d_silu_bwd_part_len(1, t32, CONV_DIM as u32, CONV_KW), 0, 0.0)?;
    let ashape = AttnShape {
        batch: 1,
        seq: t32,
        q_heads: Q_HEADS as u32,
        kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        rotary_dim: ROTARY,
    };
    let aw = 2 * (Q_HEADS + KV_HEADS) * HEAD_DIM;
    let (ap, adp) = (buf(rt, t * aw, s(42), 1.0)?, buf(rt, t * aw, 0, 0.0)?);
    let (qnw, knw, dqnw, dknw) = (
        buf(rt, HEAD_DIM, s(43), 0.3)?,
        buf(rt, HEAD_DIM, s(44), 0.3)?,
        buf(rt, HEAD_DIM, 0, 0.0)?,
        buf(rt, HEAD_DIM, 0, 0.0)?,
    );
    let qkpart = buf(rt, attn_qk_norm_rope_bwd_part_len(&ashape), 0, 0.0)?;
    let (alog, dtb, dalog, ddtb) = (
        buf(rt, GDN_HEADS, s(45), 0.5)?,
        buf(rt, GDN_HEADS, s(46), 1.0)?,
        buf(rt, GDN_HEADS, 0, 0.0)?,
        buf(rt, GDN_HEADS, 0, 0.0)?,
    );
    let gtpart = buf(rt, gdn_gates_bwd_part_len(t32, GDN_HEADS as u32), 0, 0.0)?;
    let (tdg, tdbeta) = (buf(rt, t * GDN_HEADS, s(47), 1.0)?, buf(rt, t * GDN_HEADS, s(48), 1.0)?);
    let ids: Vec<u32> = (0..t32).map(|i| (i * 104_729) % VOCAB as u32).collect();
    let ews = EmbedBwdWorkspace::new(rt, t32)?;
    let logits = GdnGateLogits {
        buf: &proj,
        ld: width as u32,
        a_off: (CONV_DIM + vd + GDN_HEADS) as u32,
        b_off: (CONV_DIM + vd) as u32,
    };
    let gparams = GdnParams {
        a_log: &alog,
        dt_bias: &dtb,
    };

    let mut ops: Vec<Op<'_>> = vec![
        Op {
            name: "gdn_train forward",
            run: Box::new(|| gdn_train_forward(rt, gd, inputs(), &o, None, &ckpt)),
            outputs: vec![("o", &o.buffer, t * vd)],
            reps: REPS,
        },
        Op {
            name: "gdn_train backward",
            run: Box::new(|| {
                gdn_train_backward(
                    rt,
                    gd,
                    inputs(),
                    &ckpt,
                    &d_o,
                    None,
                    &gws,
                    GdnTrainGrads {
                        dq: &dq,
                        dk: &dk,
                        dv: &dv,
                        dg: &dg,
                        dbeta: &dbeta,
                        ds0: None,
                    },
                )
            }),
            outputs: vec![
                ("dq", &dq.buffer, t * GDN_HEADS * GDN_DK),
                ("dv", &dv.buffer, t * vd),
                ("dg", &dg.buffer, t * GDN_HEADS),
            ],
            reps: REPS,
        },
        Op {
            name: "attn_train forward",
            run: Box::new(|| attn_train_forward(rt, &ad, &aq, &ak, &av, &ao, &lse, &aws)),
            outputs: vec![("o", &ao, t * qd), ("lse", &lse, ad.lse_len())],
            reps: REPS,
        },
        Op {
            name: "attn_train backward",
            run: Box::new(|| {
                attn_train_backward(
                    rt,
                    &ad,
                    &aq,
                    &ak,
                    &av,
                    &ao,
                    &lse,
                    &ado,
                    &AttnTrainGrads {
                        dq: &adq,
                        dk: &adk,
                        dv: &adv,
                    },
                    &aws,
                )
            }),
            outputs: vec![("dq", &adq, t * qd), ("dk", &adk, t * kvd), ("dv", &adv, t * kvd)],
            reps: REPS,
        },
        Op {
            name: "cross-entropy + grads",
            run: Box::new(|| {
                cross_entropy_rows(
                    rt,
                    CeHidden { rows: &h, off: 0 },
                    &emb,
                    &rows,
                    &targets,
                    Reduction::Mean,
                    operands,
                    &ce_ws,
                    Some(CeGrads {
                        dh: &dh,
                        dw: &dw,
                        scale: 1.0,
                    }),
                )
                .map(|_| ())
            }),
            outputs: vec![
                ("dh", &dh.buffer, (t - 1) * HIDDEN),
                ("dw (first 1024 rows)", &dw.buffer, 1024 * HIDDEN),
            ],
            reps: 1,
        },
        Op {
            name: "rms_norm_bwd",
            run: Box::new(|| rms_norm_bwd(rt, &x, &nw, &dy_h, &dx_h, &dnw, &npart, t32, HIDDEN as u32, EPS, false)),
            outputs: vec![("dx", &dx_h, t * HIDDEN), ("dw", &dnw, HIDDEN)],
            reps: REPS,
        },
        Op {
            name: "gated_rms_norm_bwd",
            run: Box::new(|| {
                gated_rms_norm_bwd(
                    rt,
                    Cols::dense(&go, vd as u32),
                    Cols {
                        buf: &proj,
                        ld: width as u32,
                        off: CONV_DIM as u32,
                    },
                    &gnw,
                    Cols::dense(&gdy, vd as u32),
                    Cols::dense(&gdo, vd as u32),
                    Cols {
                        buf: &dproj,
                        ld: width as u32,
                        off: CONV_DIM as u32,
                    },
                    &gdnw,
                    &gpart,
                    t32,
                    GDN_HEADS as u32,
                    GDN_DV as u32,
                    EPS,
                )
            }),
            outputs: vec![("dx", &gdo, t * vd), ("dw", &gdnw, GDN_DV)],
            reps: REPS,
        },
        Op {
            name: "swiglu_bwd",
            run: Box::new(|| {
                swiglu_bwd(
                    rt,
                    Cols::dense(&mg, INTER as u32),
                    Cols::dense(&mu, INTER as u32),
                    Cols::dense(&mdy, INTER as u32),
                    Cols::dense(&mdg, INTER as u32),
                    Cols::dense(&mdu, INTER as u32),
                    t32,
                    INTER as u32,
                )
            }),
            outputs: vec![("dgate", &mdg, t * INTER), ("dup", &mdu, t * INTER)],
            reps: REPS,
        },
        Op {
            name: "conv1d_silu_bwd",
            run: Box::new(|| {
                conv1d_silu_bwd(
                    rt,
                    Cols {
                        buf: &proj,
                        ld: width as u32,
                        off: 0,
                    },
                    &cw,
                    CONV_KW,
                    Cols::dense(&cdy, CONV_DIM as u32),
                    Cols {
                        buf: &dproj,
                        ld: width as u32,
                        off: 0,
                    },
                    &cdw,
                    &cpart,
                    1,
                    t32,
                    CONV_DIM as u32,
                )
            }),
            outputs: vec![("dw", &cdw, CONV_DIM * CONV_KW as usize)],
            reps: REPS,
        },
        Op {
            name: "attn_qk_norm_rope_bwd",
            run: Box::new(|| {
                attn_qk_norm_rope_bwd(
                    rt,
                    &ashape,
                    Cols::dense(&ap, aw as u32),
                    &qnw,
                    &knw,
                    &AttnQkvGrads {
                        dq: &adq,
                        dk: &adk,
                        dv: &adv,
                    },
                    &adp,
                    &dqnw,
                    &dknw,
                    &qkpart,
                    THETA,
                    EPS,
                )
            }),
            outputs: vec![("dq_norm_w", &dqnw, HEAD_DIM), ("dk_norm_w", &dknw, HEAD_DIM)],
            reps: REPS,
        },
        Op {
            name: "gdn_gates_bwd",
            run: Box::new(|| {
                gdn_gates_bwd(
                    rt,
                    &logits,
                    &gparams,
                    &tdg,
                    &tdbeta,
                    &dproj,
                    &dalog,
                    &ddtb,
                    &gtpart,
                    t32,
                    GDN_HEADS as u32,
                )
            }),
            outputs: vec![("dA_log", &dalog, GDN_HEADS), ("ddt_bias", &ddtb, GDN_HEADS)],
            reps: REPS,
        },
        Op {
            name: "embed_rows_bwd",
            run: Box::new(|| embed_rows_bwd(rt, &ids, &dx_h, &dw.buffer, VOCAB as u32, HIDDEN as u32, &ews)),
            outputs: vec![],
            reps: 1,
        },
    ];
    for op in &mut ops {
        gate(rt, op)?;
    }
    // The embedding backward adds into dW, so a poisoned dW cannot show it
    // wrote: from zeros, the rows it read must have moved.
    dw.buffer.zero();
    embed_rows_bwd(rt, &ids, &dx_h, &dw.buffer, VOCAB as u32, HIDDEN as u32, &ews)?;
    rt.synchronize()?;
    let all = dw.buffer.read_f32();
    if let Some(&id) = ids
        .iter()
        .find(|&&id| all[id as usize * HIDDEN..][..HIDDEN].iter().all(|&x| x == 0.0))
    {
        return Err(format!("embed_rows_bwd: row {id} was not written"));
    }
    println!("T = {t}: gate passed ({} ops wrote finite outputs)", ops.len());
    if check_only {
        return Ok(());
    }
    println!("{:<24} {:>10}", "op", "ms / call");
    let mut times = Vec::new();
    for op in &mut ops {
        let ms = time_ms(rt, op.reps, &mut op.run)?;
        println!("{:<24} {:>10.3}", op.name, ms);
        times.push((op.name, ms));
    }
    let get = |n: &str| times.iter().find(|(m, _)| *m == n).map(|(_, v)| *v).unwrap();
    println!(
        "GDN core, forward + backward: {:.2} ms per layer; attention core: {:.2} ms per layer",
        get("gdn_train forward") + get("gdn_train backward"),
        get("attn_train forward") + get("attn_train backward")
    );
    Ok(())
}

/// The real 2B checkpoint, in f32.
fn load_2b(rt: &Arc<GpuRuntime>) -> Res<Qwen35Model> {
    let path = std::env::var("QWEN35_2B_SAFETENSORS")
        .map_err(|_| "--step and --batch need QWEN35_2B_SAFETENSORS (the Qwen3.5-2B-Base .safetensors)".to_string())?;
    let st = SafeTensors::open(std::path::Path::new(&path))?;
    Qwen35Model::load(
        rt,
        &st,
        "model.language_model.",
        Qwen35Config::qwen35_2b()?,
        Precision::F32,
    )
}

/// Host waits, commits, dispatches and fresh allocations `f` made, and its
/// wall time.
fn traced<T>(f: impl FnOnce() -> Res<T>) -> Res<(T, tessl::infer_trace::Snapshot, f64)> {
    tessl::infer_trace::set_enabled(true);
    let (t0, s0) = (Instant::now(), tessl::infer_trace::snapshot());
    let out = f();
    let (secs, s1) = (t0.elapsed().as_secs_f64(), tessl::infer_trace::snapshot());
    tessl::infer_trace::set_enabled(false);
    let d = tessl::infer_trace::Snapshot {
        dispatches: s1.dispatches - s0.dispatches,
        barriers: s1.barriers - s0.barriers,
        commits: s1.commits - s0.commits,
        cold_allocs: s1.cold_allocs - s0.cold_allocs,
        sync_wait_us: s1.sync_wait_us - s0.sync_wait_us,
        sync_waits: s1.sync_waits - s0.sync_waits,
        residency_flushes: s1.residency_flushes - s0.residency_flushes,
    };
    Ok((out?, d, secs))
}

fn trace_line(s: &tessl::infer_trace::Snapshot) -> String {
    format!(
        "{} waits ({:.3} s blocked), {} commits, {} dispatches, {} cold allocations",
        s.sync_waits,
        s.sync_wait_us as f64 / 1e6,
        s.commits,
        s.dispatches,
        s.cold_allocs
    )
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn bench_step(rt: &Arc<GpuRuntime>, model: &Qwen35Model, tokens: usize, operands: GemmOperands) -> Res<()> {
    let ids: Vec<u32> = (0..tokens as u32).map(|i| (i * 104_729 + 17) % VOCAB as u32).collect();
    let base = rt.current_allocated_bytes();
    rt.reset_peak_allocated_bytes();
    // Only the loss is kept: holding the warm-up step would double the
    // gradients resident while timing.
    let first = model.train_step(&ids, operands)?.loss;
    if !first.is_finite() {
        return Err(format!("train_step: loss {first} is not finite"));
    }
    let mut samples = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        let step = model.train_step(&ids, operands)?;
        samples.push(t0.elapsed().as_secs_f64());
        if step.loss.to_bits() != first.to_bits() {
            return Err("train_step: the loss changed between identical steps".into());
        }
    }
    let secs = median(samples);
    println!(
        "train_step, T = {tokens}: {:.3} s ({:.0} tokens/s), loss {:.4}",
        secs,
        tokens as f64 / secs,
        first
    );
    let peak = rt.peak_allocated_bytes();
    println!(
        "  device peak over the steps: {:.2} GiB ({:.2} GiB above the loaded model)",
        gib(peak),
        gib(peak.saturating_sub(base))
    );
    let (step, trace, secs) = traced(|| model.train_step(&ids, operands))?;
    if step.loss.to_bits() != first.to_bits() {
        return Err("train_step: the traced step's loss differs".into());
    }
    drop(step);
    println!("  one traced step: {secs:.3} s, {}", trace_line(&trace));
    Ok(())
}

/// One clipped optimizer step as a trainer runs it: the step's gradients
/// into a bank, the bank's norm, then AdamW with the clip coefficient.
fn bench_clip(model: &Qwen35Model, tokens: usize, operands: GemmOperands) -> Res<()> {
    let ids: Vec<u32> = (0..tokens as u32).map(|i| (i * 104_729 + 17) % VOCAB as u32).collect();
    let bank = Qwen35Grads::zeros_like(model)?;
    let mut state = AdamW::new(model)?;
    let wd = model.default_weight_decay(0.01)?;
    let mut clip = || -> Res<[(f64, tessl::infer_trace::Snapshot); 3]> {
        let (loss, a, ta) =
            traced(|| model.train_step_into(&ids, operands, tessl::qwen35_train::Supervise::Causal, &bank, false))?;
        if !loss.is_finite() {
            return Err(format!("--clip: loss {loss} is not finite"));
        }
        let (sq, b, tb) = traced(|| model.grad_sq_norm(&bank))?;
        if !(sq.is_finite() && sq > 0.0) {
            return Err(format!("--clip: gradient norm^2 {sq}"));
        }
        let hyper = AdamWHyper {
            lr: 1e-9,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_scale: (1.0 / sq.sqrt().max(1.0)) as f32 as f64,
        };
        let ((), c, tc) = traced(|| model.adamw_step(&bank, &mut state, &hyper, &wd))?;
        Ok([(ta, a), (tb, b), (tc, c)])
    };
    clip()?;
    let mut runs = Vec::new();
    for _ in 0..3 {
        runs.push(clip()?);
    }
    println!("clipped optimizer step, T = {tokens} (median of 3; waits from the last run):");
    for (i, name) in ["train_step_into", "grad_sq_norm", "adamw_step"].iter().enumerate() {
        let ms = median(runs.iter().map(|r| r[i].0 * 1e3).collect());
        println!("  {name:<16} {ms:>10.1} ms, {}", trace_line(&runs[2][i].1));
    }
    let total = median(runs.iter().map(|r| r.iter().map(|x| x.0).sum::<f64>()).collect());
    let waits: u64 = runs[2].iter().map(|x| x.1.sync_waits).sum();
    println!("  total            {:>10.1} ms, {waits} waits", total * 1e3);
    Ok(())
}

/// One `--batch` shape: `rows` sequences of `len` tokens; `span` of them (the
/// last ones) take a loss outside tessl instead of a supervised position.
#[derive(Clone, Copy, Debug)]
struct BatchShape {
    rows: usize,
    len: usize,
    span: usize,
}

fn parse_batch(v: &str) -> Res<BatchShape> {
    let bad = || format!("--batch expects ROWS,LEN or ROWS,LEN,SPAN_ROWS, got {v:?}");
    let parts = v
        .split(',')
        .map(|x| x.parse::<usize>().map_err(|_| bad()))
        .collect::<Res<Vec<_>>>()?;
    let (rows, len, span) = match parts[..] {
        [r, l] => (r, l, 0),
        [r, l, s] => (r, l, s),
        _ => return Err(bad()),
    };
    // A span row hands out 4 distinct positions (0, len/3, len/2, len - 2).
    if rows == 0 || len < 2 || span > rows || (span > 0 && len < 5) {
        return Err(format!(
            "--batch {v:?}: need rows >= 1, len >= 2 (5 with span rows), span rows <= rows"
        ));
    }
    Ok(BatchShape { rows, len, span })
}

/// One optimizer step's gradients for a batch run row by row, as a padded
/// batch runs through tessl: each row's forward and backward into one bank,
/// accumulated. A letter row supervises one position at `1 / letter rows`;
/// a span row scores nothing in tessl, hands 4 hidden rows out and takes a
/// fixed gradient back for them. Median of 3 after one warm-up; per row is
/// the batch time over the rows. The AdamW step itself is not included.
fn bench_batch(rt: &Arc<GpuRuntime>, model: &Qwen35Model, b: BatchShape, operands: GemmOperands) -> Res<()> {
    let letters = b.rows - b.span;
    let rows: Vec<Vec<u32>> = (0..b.rows)
        .map(|r| {
            (0..b.len as u32)
                .map(|i| ((i + 31 * r as u32) * 104_729 + 17) % VOCAB as u32)
                .collect()
        })
        .collect();
    let at: Vec<u32> = [0, b.len / 3, b.len / 2, b.len - 2].iter().map(|&p| p as u32).collect();
    let dh = tensor(rt, &[at.len(), HIDDEN], 7, 1e-3)?;
    let hidden = rt.alloc_tensor_f32(&[at.len(), HIDDEN])?;
    let bank = tessl::qwen35_train::Qwen35Grads::zeros_like(model)?;
    let row = |i: usize| -> Res<()> {
        let (ids, accumulate) = (&rows[i], i > 0);
        if i < letters {
            let (pos, tgt) = ([b.len as u32 - 2], [ids[b.len - 1]]);
            let sup = tessl::qwen35_train::Supervise::Rows {
                positions: &pos,
                targets: &tgt,
                scale: 1.0 / letters as f32,
            };
            model.train_step_into(ids, operands, sup, &bank, accumulate)?;
        } else {
            let sup = tessl::qwen35_train::Supervise::Rows {
                positions: &[],
                targets: &[],
                scale: 1.0,
            };
            let p = model.train_forward(ids, operands, sup)?;
            p.hidden(&at, &hidden)?;
            model.train_backward_into(p, Some((&at, &dh)), &bank, accumulate)?;
        }
        Ok(())
    };
    let run = || -> Res<()> {
        for i in 0..b.rows {
            row(i)?;
        }
        rt.synchronize()
    };
    run()?;
    // The warm-up's gradients must be there and finite before anything is timed.
    let sq = model.grad_sq_norm(&bank)?;
    if !(sq.is_finite() && sq > 0.0) {
        return Err(format!("--batch: the warm-up left a gradient norm^2 of {sq}"));
    }
    let mut samples = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        run()?;
        samples.push(t0.elapsed().as_secs_f64());
    }
    let secs = median(samples);
    println!(
        "batch {} rows x {} tokens ({} span): {:.3} s per optimizer step, {:.3} s per row ({:.0} tokens/s)",
        b.rows,
        b.len,
        b.span,
        secs,
        secs / b.rows as f64,
        (b.rows * b.len) as f64 / secs
    );
    // One more run, untimed above, each row waited for and traced: where a
    // batch's time goes row by row (wall time, time blocked on the GPU,
    // buffer allocations, pool hits included, and command buffers committed).
    tessl::infer_trace::set_enabled(true);
    for i in 0..b.rows {
        let (t0, s0) = (Instant::now(), tessl::infer_trace::snapshot());
        row(i)?;
        rt.synchronize()?;
        let (secs, s1) = (t0.elapsed().as_secs_f64(), tessl::infer_trace::snapshot());
        println!(
            "  row {i} ({}{}): {secs:.3} s, {:.3} s waiting on the GPU, {} allocations, {} commits",
            if i < letters { "letter" } else { "span" },
            if i > 0 { ", accumulated" } else { "" },
            (s1.sync_wait_us - s0.sync_wait_us) as f64 / 1e6,
            s1.cold_allocs - s0.cold_allocs,
            s1.commits - s0.commits,
        );
    }
    tessl::infer_trace::set_enabled(false);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut check_only, mut step, mut ts, mut operands) = (false, None, Vec::new(), GemmOperands::ExactF32);
    let (mut batches, mut step_only, mut async_encode, mut clip) = (Vec::new(), false, false, None);
    for arg in std::env::args().skip(1) {
        if arg == "--bf16" {
            operands = GemmOperands::Bf16;
        } else if arg == "--check-only" {
            check_only = true;
        } else if arg == "--step-only" {
            step_only = true;
        } else if arg == "--async" {
            async_encode = true;
        } else if let Some(n) = arg.strip_prefix("--clip=") {
            clip = Some(
                n.parse::<usize>()
                    .map_err(|_| format!("--clip expects a token count, got {n:?}"))?,
            );
        } else if let Some(v) = arg.strip_prefix("--batch=") {
            batches.push(parse_batch(v)?);
        } else if let Some(n) = arg.strip_prefix("--step=") {
            step = Some(
                n.parse::<usize>()
                    .map_err(|_| format!("--step expects a token count, got {n:?}"))?,
            );
        } else {
            let t: usize = arg.parse().map_err(|_| {
                format!(
                    "expected a token count, --check-only, --step-only, --bf16, --async, --step=N, --clip=N \
                     or --batch=B,L[,S], got {arg:?}"
                )
            })?;
            if t < 2 {
                return Err("T must be at least 2 (one prediction)".into());
            }
            ts.push(t);
        }
    }
    // The per-op gate runs first whatever else is asked: no timing without it.
    if ts.is_empty() {
        ts.push(2048);
    }
    let rt = GpuRuntime::new()?;
    if rt.relaxed_precision() {
        return Err("the training path needs exact-f32 GEMMs; switch relaxed precision off".into());
    }
    println!("device: {}", rt.device_name());
    for t in ts {
        bench_t(&rt, t, check_only || step_only, operands)?;
    }
    let model_runs = step.is_some() || clip.is_some() || !batches.is_empty();
    if check_only || !model_runs {
        if check_only && model_runs {
            println!("--check-only: the per-op gate ran; --step, --clip and --batch were not run");
        }
        return Ok(());
    }
    let model = load_2b(&rt)?;
    rt.set_async_encode(async_encode)?;
    println!(
        "encode: {}; TESSL_MID_COMMIT={}",
        if async_encode {
            "async (--async)"
        } else {
            "every dispatch waited"
        },
        std::env::var("TESSL_MID_COMMIT").unwrap_or_else(|_| "unset".into())
    );
    if let Some(n) = step {
        bench_step(&rt, &model, n, operands)?;
    }
    if let Some(n) = clip {
        bench_clip(&model, n, operands)?;
    }
    for b in batches {
        bench_batch(&rt, &model, b, operands)?;
    }
    Ok(())
}
