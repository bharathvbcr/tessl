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

use common::{buf, random_f32, seeded, with_gpu};
use tessl::qwen35::{self, Cols, OutCols};
use tessl::qwen35_bwd::{gated_rms_norm_bwd, gated_rms_norm_bwd_part_len, rms_norm_bwd, rms_norm_bwd_part_len};
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
    with_gpu(|rt| {
        let segs = collect(rt);
        for (name, v) in &segs {
            assert!(v.iter().all(|f| f.is_finite()), "{name}: non-finite value");
        }
        let bytes = to_bytes(&segs);
        if let Ok(path) = std::env::var("RSTD_BITS_OUT") {
            fs::write(&path, &bytes).unwrap();
            eprintln!("recorded {} bytes to {path}", bytes.len());
        }
        if let Ok(path) = std::env::var("RSTD_BITS_BASE") {
            let base = fs::read(&path).unwrap();
            assert_eq!(base.len(), bytes.len(), "baseline {path} has a different shape");
            let mut at = 0usize;
            for (name, v) in &segs {
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
    });
}
