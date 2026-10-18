//! The Qwen3.5 RMSNorm forward and backward compute the same `rstd`.
//!
//! Training rebuilds `rstd = rsqrt(mean(x^2) + eps)` in the backward from `x`
//! instead of saving it, so the forward's `rstd` and the backward's must be the
//! same bits (owner decision on card ft-1a0f8c18, AC5: both are
//! `precise::rsqrt`). Neither kernel exposes `rstd`, so each test builds rows
//! on which it can be read out of both:
//!
//! * `x[0] = x[1] = a` with `a` a power of two, the other columns anything.
//! * `qwen35_rms_norm` (`w = 0`): `y[1] = a * rstd`. The backward with
//!   `dy[0] = a, dy[1] = -a` has `dot = a*a - a*a = 0` exactly, so
//!   `dx[0] = rstd * a`. Equal bits iff the two `rstd` are.
//! * `qwen35_gated_rms_norm` (`w = 1`, `z[0] = z[1] = 64`, where `silu(z) = 64`
//!   exactly): `y[0] = a * rstd * 64`. The backward with `dy[0] = 1, dy[1] = -1`
//!   cancels `m` exactly and gives `dx[0] = rstd * 64`, so `y[0] == a * dx[0]`.
//!
//! The rest of the file is a characterization: `characterize` collects the
//! forward outputs and the backward gradients on fixed seeded inputs (large,
//! small and zero rows, the eps edge) as f32 bits. With `RSTD_BITS_OUT=<file>`
//! it records them; with `RSTD_BITS_BASE=<file>` it prints the max ulp change
//! per tensor against an earlier recording. Without either it only checks
//! that every value is finite.

mod common;

use std::fs;
use std::sync::Arc;

use common::{buf, random_f32, seeded, with_gpu};
use tessl::qwen35::{self, AttnShape, AttnTargets, Cols, OutCols};
use tessl::qwen35_bwd::{
    attn_qk_norm_rope_bwd, attn_qk_norm_rope_bwd_part_len, gated_rms_norm_bwd, gated_rms_norm_bwd_part_len, rms_norm_bwd,
    rms_norm_bwd_part_len, AttnQkvGrads,
};
use tessl::tensor::DType;
use tessl::GpuRuntime;

const SENTINEL: f32 = -7.25e27;
const EPS: [f32; 2] = [1e-6, 1e-12];
/// Row magnitudes as powers of two: tiny (eps dominates), the eps edge for
/// `mean(x^2) ~ 1e-6` (2^-10), unit, and large.
const EXPONENTS: [i32; 14] = [-70, -62, -40, -20, -12, -10, -8, -4, 0, 0, 4, 8, 20, 30];

fn pow2(k: i32) -> f32 {
    f32::powi(2.0, k)
}

/// `rows` rows of `d` values at `EXPONENTS[r % len]`, then `x[0] = x[1] = a`.
/// A zero row comes at index 1 mod 7 for the first rows to cover `ss = 0`.
fn rows_x(rows: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut x = random_f32(rows * d, seed);
    let mut a = Vec::new();
    for r in 0..rows {
        let k = EXPONENTS[r % EXPONENTS.len()];
        let row = &mut x[r * d..(r + 1) * d];
        for v in row.iter_mut() {
            *v *= pow2(k);
        }
        if r % 7 == 1 {
            row.fill(0.0);
        }
        row[0] = pow2(k);
        row[1] = pow2(k);
        a.push(pow2(k));
    }
    (x, a)
}

fn rms_fwd(rt: &std::sync::Arc<GpuRuntime>, x: &[f32], w: &[f32], rows: usize, d: usize, eps: f32) -> Vec<f32> {
    let (xb, wb) = (buf(rt, x), buf(rt, w));
    let out = seeded(rt, rows * d, SENTINEL);
    qwen35::rms_norm(rt, &xb, &wb, &out, DType::F32, rows as u32, d as u32, eps).unwrap();
    rt.synchronize().unwrap();
    out.read_f32()[..rows * d].to_vec()
}

fn rms_bwd(
    rt: &std::sync::Arc<GpuRuntime>,
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    rows: usize,
    d: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (xb, wb, dyb) = (buf(rt, x), buf(rt, w), buf(rt, dy));
    let dx = seeded(rt, rows * d, SENTINEL);
    let dw = seeded(rt, d, SENTINEL);
    let part = seeded(rt, rms_norm_bwd_part_len(rows as u32, d as u32).max(1), SENTINEL);
    rms_norm_bwd(rt, &xb, &wb, &dyb, &dx, &dw, &part, rows as u32, d as u32, eps, false).unwrap();
    rt.synchronize().unwrap();
    (dx.read_f32()[..rows * d].to_vec(), dw.read_f32()[..d].to_vec())
}

#[allow(clippy::too_many_arguments)]
fn gated_fwd(
    rt: &std::sync::Arc<GpuRuntime>,
    x: &[f32],
    z: &[f32],
    w: &[f32],
    rows: usize,
    heads: usize,
    d: usize,
    eps: f32,
) -> Vec<f32> {
    let width = heads * d;
    let (xb, zb, wb) = (buf(rt, x), buf(rt, z), buf(rt, w));
    let out = seeded(rt, rows * width, SENTINEL);
    let oc = OutCols {
        cols: Cols::dense(&out, width as u32),
        dtype: DType::F32,
    };
    qwen35::gated_rms_norm(
        rt,
        Cols::dense(&xb, width as u32),
        Cols::dense(&zb, width as u32),
        &wb,
        oc,
        rows as u32,
        heads as u32,
        d as u32,
        eps,
    )
    .unwrap();
    rt.synchronize().unwrap();
    out.read_f32()[..rows * width].to_vec()
}

#[allow(clippy::too_many_arguments)]
fn gated_bwd(
    rt: &std::sync::Arc<GpuRuntime>,
    x: &[f32],
    z: &[f32],
    w: &[f32],
    dy: &[f32],
    rows: usize,
    heads: usize,
    d: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let width = heads * d;
    let (xb, zb, wb, dyb) = (buf(rt, x), buf(rt, z), buf(rt, w), buf(rt, dy));
    let dx = seeded(rt, rows * width, SENTINEL);
    let dz = seeded(rt, rows * width, SENTINEL);
    let dw = seeded(rt, d, SENTINEL);
    let part = seeded(
        rt,
        gated_rms_norm_bwd_part_len(rows as u32, heads as u32, d as u32).max(1),
        SENTINEL,
    );
    let dense = |b| Cols::dense(b, width as u32);
    gated_rms_norm_bwd(
        rt,
        dense(&xb),
        dense(&zb),
        &wb,
        dense(&dyb),
        dense(&dx),
        dense(&dz),
        &dw,
        &part,
        rows as u32,
        heads as u32,
        d as u32,
        eps,
    )
    .unwrap();
    rt.synchronize().unwrap();
    (
        dx.read_f32()[..rows * width].to_vec(),
        dz.read_f32()[..rows * width].to_vec(),
        dw.read_f32()[..d].to_vec(),
    )
}

/// Rows whose forward `rstd` and backward `rstd` differ in bits, as messages.
fn rms_mismatches(rt: &std::sync::Arc<GpuRuntime>, d: usize, eps: f32) -> Vec<String> {
    let rows = 28;
    let (x, a) = rows_x(rows, d, 7100);
    let w = vec![0.0f32; d];
    let y = rms_fwd(rt, &x, &w, rows, d, eps);
    let mut dy = vec![0.0f32; rows * d];
    for r in 0..rows {
        dy[r * d] = a[r];
        dy[r * d + 1] = -a[r];
    }
    let (dx, _) = rms_bwd(rt, &x, &w, &dy, rows, d, eps);
    (0..rows)
        .filter(|&r| y[r * d + 1].to_bits() != dx[r * d].to_bits())
        .map(|r| {
            format!(
                "rms d={d} eps={eps:e} row {r} a={:e}: forward a*rstd {:e} ({:#010x}), backward {:e} ({:#010x})",
                a[r],
                y[r * d + 1],
                y[r * d + 1].to_bits(),
                dx[r * d],
                dx[r * d].to_bits()
            )
        })
        .collect()
}

fn gated_mismatches(rt: &std::sync::Arc<GpuRuntime>, d: usize, eps: f32) -> Vec<String> {
    let rows = 28;
    let (x, a) = rows_x(rows, d, 7200);
    let mut z = random_f32(rows * d, 7201);
    z.iter_mut().for_each(|v| *v *= 3.0);
    // `silu(64) = 64 * sigmoid(64) = 64` exactly (the sigmoid is 1 to f32), so
    // `s` is a power of two and the products below do not round: the kernels
    // may associate `a * rstd * s` either way (fast math does).
    for r in 0..rows {
        z[r * d] = 64.0;
        z[r * d + 1] = 64.0;
    }
    let w = vec![1.0f32; d];
    let y = gated_fwd(rt, &x, &z, &w, rows, 1, d, eps);
    let mut dy = vec![0.0f32; rows * d];
    for r in 0..rows {
        dy[r * d] = 1.0;
        dy[r * d + 1] = -1.0;
    }
    let (dx, _, _) = gated_bwd(rt, &x, &z, &w, &dy, rows, 1, d, eps);
    (0..rows)
        .filter(|&r| y[r * d].to_bits() != (a[r] * dx[r * d]).to_bits())
        .map(|r| {
            format!(
                "gated d={d} eps={eps:e} row {r} a={:e}: forward {:e} ({:#010x}), a * backward {:e} ({:#010x})",
                a[r],
                y[r * d],
                y[r * d].to_bits(),
                a[r] * dx[r * d],
                (a[r] * dx[r * d]).to_bits()
            )
        })
        .collect()
}

#[test]
fn rms_norm_backward_rebuilds_the_forward_rstd() {
    with_gpu(|rt| {
        let mut bad = Vec::new();
        for d in [128usize, 300] {
            for eps in EPS {
                bad.extend(rms_mismatches(rt, d, eps));
            }
        }
        assert!(bad.is_empty(), "{} rows differ:\n{}", bad.len(), bad.join("\n"));
    });
}

#[test]
fn gated_rms_norm_backward_rebuilds_the_forward_rstd() {
    with_gpu(|rt| {
        let mut bad = Vec::new();
        for d in [64usize, 128] {
            for eps in EPS {
                bad.extend(gated_mismatches(rt, d, eps));
            }
        }
        assert!(bad.is_empty(), "{} rows differ:\n{}", bad.len(), bad.join("\n"));
    });
}

// ------------------------------------------------------- characterization ---

/// Distance in f32 representable values (0 for equal bits; -0 and +0 are 1
/// apart and so are reported as 0 only when the bits match).
fn ulp_dist(a: f32, b: f32) -> u64 {
    let key = |v: f32| -> i64 {
        let b = v.to_bits();
        if b >> 31 == 1 {
            -i64::from(b & 0x7fff_ffff)
        } else {
            i64::from(b)
        }
    };
    key(a).abs_diff(key(b))
}

fn collect(rt: &std::sync::Arc<GpuRuntime>) -> Vec<(String, Vec<f32>)> {
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    for eps in EPS {
        let d = 300usize;
        let rows = 28usize;
        let (x, _) = rows_x(rows, d, 7300);
        let w: Vec<f32> = random_f32(d, 7301).iter().map(|v| 0.9 * v).collect();
        let dy = random_f32(rows * d, 7302);
        out.push((format!("rms fwd eps={eps:e}"), rms_fwd(rt, &x, &w, rows, d, eps)));
        let (dx, dw) = rms_bwd(rt, &x, &w, &dy, rows, d, eps);
        out.push((format!("rms bwd dx eps={eps:e}"), dx));
        out.push((format!("rms bwd dw eps={eps:e}"), dw));

        let (heads, d) = (2usize, 128usize);
        let width = heads * d;
        // `rows_x` perturbs only each row's first two columns of `width`.
        let (x, _) = rows_x(rows, width, 7310);
        let z: Vec<f32> = random_f32(rows * width, 7311).iter().map(|v| 3.0 * v).collect();
        let w: Vec<f32> = random_f32(d, 7312).iter().map(|v| 1.0 + 0.2 * v).collect();
        let dy = random_f32(rows * width, 7313);
        out.push((
            format!("gated fwd eps={eps:e}"),
            gated_fwd(rt, &x, &z, &w, rows, heads, d, eps),
        ));
        let (dx, dz, dw) = gated_bwd(rt, &x, &z, &w, &dy, rows, heads, d, eps);
        out.push((format!("gated bwd dx eps={eps:e}"), dx));
        out.push((format!("gated bwd dz eps={eps:e}"), dz));
        out.push((format!("gated bwd dw eps={eps:e}"), dw));
    }
    out
}

fn to_bytes(segs: &[(String, Vec<f32>)]) -> Vec<u8> {
    segs.iter()
        .flat_map(|(_, v)| v.iter().flat_map(|f| f.to_bits().to_le_bytes()))
        .collect()
}

#[test]
fn characterize() {
    with_gpu(|rt| record_or_compare(&collect(rt), "RSTD_BITS_OUT", "RSTD_BITS_BASE"));
}

/// The qk-norm pair's characterization, recorded apart (`RSTD_QK_OUT`,
/// `RSTD_QK_BASE`) so each baseline keeps its own shape.
#[test]
fn characterize_qk() {
    with_gpu(|rt| record_or_compare(&collect_qk(rt), "RSTD_QK_OUT", "RSTD_QK_BASE"));
}

fn record_or_compare(segs: &[(String, Vec<f32>)], out_var: &str, base_var: &str) {
    {
        for (name, v) in segs {
            assert!(v.iter().all(|f| f.is_finite()), "{name}: non-finite value");
        }
        let bytes = to_bytes(segs);
        if let Ok(path) = std::env::var(out_var) {
            fs::write(&path, &bytes).unwrap();
            eprintln!("recorded {} bytes to {path}", bytes.len());
        }
        if let Ok(path) = std::env::var(base_var) {
            let base = fs::read(&path).unwrap();
            assert_eq!(base.len(), bytes.len(), "baseline {path} has a different shape");
            let mut at = 0usize;
            for (name, v) in segs {
                let (mut max, mut differ, mut worst) = (0u64, 0usize, 0usize);
                for (i, f) in v.iter().enumerate() {
                    let o = u32::from_le_bytes(base[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
                    let dist = ulp_dist(f32::from_bits(o), *f);
                    differ += usize::from(dist != 0);
                    if dist > max {
                        (max, worst) = (dist, i);
                    }
                }
                at += 4 * v.len();
                eprintln!(
                    "ULP {name}: max {max} ulp at {worst}, {differ}/{} elements differ",
                    v.len()
                );
            }
        }
    }
}

// ------------------------------------------------------------ qk norm + rope ---

const QK_D: usize = 64;
const QK_T: usize = 28;

fn qk_shape(rot: usize) -> AttnShape {
    AttnShape {
        batch: 1,
        seq: QK_T as u32,
        q_heads: 1,
        kv_heads: 1,
        head_dim: QK_D as u32,
        rotary_dim: rot as u32,
    }
}

/// Forward over `proj` (`[QK_T, 4 * QK_D]`: q and its gate, k, v): the
/// rotated queries `[QK_T, QK_D]` and the K cache `[QK_T, QK_D]`.
fn qk_fwd(rt: &Arc<GpuRuntime>, proj: &[f32], qw: &[f32], kw: &[f32], rot: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let (pb, qwb, kwb) = (buf(rt, proj), buf(rt, qw), buf(rt, kw));
    let q = seeded(rt, QK_T * QK_D, SENTINEL);
    let k = seeded(rt, QK_T * QK_D, SENTINEL);
    let v = seeded(rt, QK_T * QK_D, SENTINEL);
    let targets = AttnTargets {
        q_out: &q,
        k_cache: &k,
        v_cache: &v,
    };
    qwen35::attn_qk_norm_rope(
        rt,
        &qk_shape(rot),
        Cols::dense(&pb, (4 * QK_D) as u32),
        &qwb,
        &kwb,
        &targets,
        0,
        1e6,
        eps,
    )
    .unwrap();
    rt.synchronize().unwrap();
    (q.read_f32()[..QK_T * QK_D].to_vec(), k.read_f32()[..QK_T * QK_D].to_vec())
}

/// Backward: `dproj` (`[QK_T, 4 * QK_D]`) and the two norm-weight gradients.
#[allow(clippy::too_many_arguments)]
fn qk_bwd(
    rt: &Arc<GpuRuntime>,
    proj: &[f32],
    qw: &[f32],
    kw: &[f32],
    dq: &[f32],
    dk: &[f32],
    rot: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let shape = qk_shape(rot);
    let (pb, qwb, kwb) = (buf(rt, proj), buf(rt, qw), buf(rt, kw));
    let (dqb, dkb) = (buf(rt, dq), buf(rt, dk));
    let dvb = seeded(rt, QK_T * QK_D, 0.0);
    let dp = seeded(rt, QK_T * 4 * QK_D, SENTINEL);
    let (dqw, dkw) = (seeded(rt, QK_D, SENTINEL), seeded(rt, QK_D, SENTINEL));
    let part = seeded(rt, attn_qk_norm_rope_bwd_part_len(&shape).max(1), SENTINEL);
    let grads = AttnQkvGrads {
        dq: &dqb,
        dk: &dkb,
        dv: &dvb,
    };
    attn_qk_norm_rope_bwd(
        rt,
        &shape,
        Cols::dense(&pb, (4 * QK_D) as u32),
        &qwb,
        &kwb,
        &grads,
        &dp,
        &dqw,
        &dkw,
        &part,
        1e6,
        eps,
    )
    .unwrap();
    rt.synchronize().unwrap();
    (
        dp.read_f32()[..QK_T * 4 * QK_D].to_vec(),
        dqw.read_f32()[..QK_D].to_vec(),
        dkw.read_f32()[..QK_D].to_vec(),
    )
}

/// `[QK_T, 4 * QK_D]` with each row at `EXPONENTS[r % len]`, its q head
/// (columns `0..D`) and k head (columns `2D..3D`) starting `a, a`.
fn qk_proj(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let w = 4 * QK_D;
    let (mut p, mut a) = (random_f32(QK_T * w, seed), Vec::new());
    for r in 0..QK_T {
        let k = EXPONENTS[r % EXPONENTS.len()];
        for v in &mut p[r * w..(r + 1) * w] {
            *v *= pow2(k);
        }
        for col in [0, 2 * QK_D] {
            if r % 7 == 1 {
                p[r * w + col..r * w + col + QK_D].fill(0.0);
            }
            p[r * w + col] = pow2(k);
            p[r * w + col + 1] = pow2(k);
        }
        a.push(pow2(k));
    }
    (p, a)
}

/// Rows whose forward `rstd` and backward `rstd` differ in bits, for the q and
/// the k head. With `rotary_dim = 0` and `w = 0` the observation is the rms
/// norm's: forward `y[1] = a * rstd`, backward (`dy = [a, -a, 0, ...]`)
/// `dx[0] = rstd * a`.
fn qk_mismatches(rt: &Arc<GpuRuntime>, eps: f32) -> Vec<String> {
    let (p, a) = qk_proj(7400);
    let w = vec![0.0f32; QK_D];
    let (q, k) = qk_fwd(rt, &p, &w, &w, 0, eps);
    let (mut dq, mut dk) = (vec![0.0f32; QK_T * QK_D], vec![0.0f32; QK_T * QK_D]);
    for r in 0..QK_T {
        for g in [&mut dq, &mut dk] {
            g[r * QK_D] = a[r];
            g[r * QK_D + 1] = -a[r];
        }
    }
    let (dp, _, _) = qk_bwd(rt, &p, &w, &w, &dq, &dk, 0, eps);
    let mut bad = Vec::new();
    for r in 0..QK_T {
        for (what, fwd, bwd) in [
            ("q", q[r * QK_D + 1], dp[r * 4 * QK_D]),
            ("k", k[r * QK_D + 1], dp[r * 4 * QK_D + 2 * QK_D]),
        ] {
            if fwd.to_bits() != bwd.to_bits() {
                bad.push(format!(
                    "qk {what} eps={eps:e} row {r} a={:e}: forward {fwd:e} ({:#010x}), backward {bwd:e} ({:#010x})",
                    a[r],
                    fwd.to_bits(),
                    bwd.to_bits()
                ));
            }
        }
    }
    bad
}

#[test]
fn qk_norm_rope_backward_rebuilds_the_forward_rstd() {
    with_gpu(|rt| {
        let bad: Vec<String> = EPS.iter().flat_map(|&e| qk_mismatches(rt, e)).collect();
        assert!(bad.is_empty(), "{} rows differ:\n{}", bad.len(), bad.join("\n"));
    });
}

/// Forward outputs through both entries (Qwen's `1 + w` and the packed
/// `weight_bias = 0` EmbeddingGemma 2 reads) and the backward's gradients, on
/// seeded rows of every magnitude, with a real rotary slice.
fn collect_qk(rt: &Arc<GpuRuntime>) -> Vec<(String, Vec<f32>)> {
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    for eps in EPS {
        let (p, _) = qk_proj(7500);
        let qw: Vec<f32> = random_f32(QK_D, 7501).iter().map(|v| 0.9 * v).collect();
        let kw: Vec<f32> = random_f32(QK_D, 7502).iter().map(|v| 0.9 * v).collect();
        let (q, k) = qk_fwd(rt, &p, &qw, &kw, 16, eps);
        out.push((format!("qk fwd q eps={eps:e}"), q));
        out.push((format!("qk fwd k eps={eps:e}"), k));
        let dq = random_f32(QK_T * QK_D, 7503);
        let dk = random_f32(QK_T * QK_D, 7504);
        let (dp, dqw, dkw) = qk_bwd(rt, &p, &qw, &kw, &dq, &dk, 16, eps);
        out.push((format!("qk bwd dproj eps={eps:e}"), dp));
        out.push((format!("qk bwd dq_norm_w eps={eps:e}"), dqw));
        out.push((format!("qk bwd dk_norm_w eps={eps:e}"), dkw));

        // The packed entry with `weight_bias = 0`: `* w`, as EmbeddingGemma 2.
        let (pb, qwb, kwb) = (buf(rt, &p), buf(rt, &qw), buf(rt, &kw));
        let q = seeded(rt, QK_T * QK_D, SENTINEL);
        let k = seeded(rt, QK_T * QK_D, SENTINEL);
        let v = seeded(rt, QK_T * QK_D, SENTINEL);
        let targets = AttnTargets {
            q_out: &q,
            k_cache: &k,
            v_cache: &v,
        };
        qwen35::attn_qk_norm_rope_packed(
            rt,
            &qk_shape(16),
            Cols::dense(&pb, (4 * QK_D) as u32),
            (2 * QK_D) as u32,
            0.0,
            &qwb,
            &kwb,
            &targets,
            0,
            1e6,
            eps,
        )
        .unwrap();
        rt.synchronize().unwrap();
        out.push((format!("qk packed w-bias 0 q eps={eps:e}"), q.read_f32()[..QK_T * QK_D].to_vec()));
        out.push((format!("qk packed w-bias 0 k eps={eps:e}"), k.read_f32()[..QK_T * QK_D].to_vec()));
    }
    out
}
