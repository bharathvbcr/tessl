//! Backward of the Qwen3.5 row-local ops (`tessl::qwen35_bwd`).
//!
//! For each op: the f64 backward below is checked against central finite
//! differences of the f64 forward (clean in f64), then the kernels against
//! that backward within `1e-4` of the largest reference magnitude, at the
//! block edges of the weight-gradient reduction and in the real layouts
//! (windows of fused projections). Weight gradients are bit-identical on a
//! rerun.

mod common;

use std::sync::Arc;

use common::qwen35::{sigmoid, silu};
use common::{buf, random_f32, seeded, with_gpu};
use tessl::qwen35::Cols;
use tessl::qwen35_bwd::{
    attn_gate_bwd, gated_rms_norm_bwd, gated_rms_norm_bwd_part_len, rms_norm_bwd, rms_norm_bwd_part_len, swiglu_bwd,
};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

const SENTINEL: f32 = -7.25e27;

fn win(buf: &GpuBuffer, ld: usize, off: usize) -> Cols<'_> {
    Cols { buf, ld: ld as u32, off: off as u32 }
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn silu_grad(g: f64) -> f64 {
    let s = sigmoid(g);
    s * (1.0 + g * (1.0 - s))
}

fn close(label: &str, got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g != SENTINEL, "{label}[{i}]: never written");
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g}");
        let e = (f64::from(g) - w).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    let rel = worst.0 / peak;
    eprintln!("{label}: {rel:.2e} of max|ref|");
    assert!(rel <= 1e-4, "{label}: max err {:.3e} at {} (got {} want {}), {rel:.2e} of {peak:.3e}", worst.0, worst.1, got[worst.1], want[worst.1]);
    rel
}

/// Central differences of `f` at `x` along each coordinate, against `grad`.
fn fd_check(name: &str, x: &[f64], grad: &[f64], f: &dyn Fn(&[f64]) -> f64) {
    let h = 1e-6;
    let mut worst = 0.0f64;
    for i in 0..x.len() {
        let (mut p, mut m) = (x.to_vec(), x.to_vec());
        p[i] += h;
        m[i] -= h;
        let fd = (f(&p) - f(&m)) / (2.0 * h);
        worst = worst.max((fd - grad[i]).abs() / (1.0 + fd.abs()));
    }
    assert!(worst < 1e-7, "{name}: worst {worst:.2e} against finite differences");
}

// ------------------------------------------------------------- RMSNorm ---

fn rms_fwd(x: &[f64], w: &[f64], d: usize, eps: f64) -> Vec<f64> {
    x.chunks(d)
        .flat_map(|r| {
            let rstd = 1.0 / (r.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
            r.iter().zip(w).map(move |(v, wv)| v * rstd * wv).collect::<Vec<_>>()
        })
        .collect()
}

fn rms_bwd(x: &[f64], w: &[f64], dy: &[f64], d: usize, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let mut dx = vec![0.0; x.len()];
    let mut dw = vec![0.0; d];
    for (r, (xr, gr)) in x.chunks(d).zip(dy.chunks(d)).enumerate() {
        let rstd = 1.0 / (xr.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
        let dot: f64 = (0..d).map(|j| gr[j] * w[j] * xr[j]).sum();
        for j in 0..d {
            dx[r * d + j] = rstd * gr[j] * w[j] - xr[j] * rstd.powi(3) * dot / d as f64;
            dw[j] += gr[j] * xr[j] * rstd;
        }
    }
    (dx, dw)
}

#[test]
fn rms_norm_reference_backward_is_the_derivative() {
    let (rows, d, eps) = (3, 7, 1e-6);
    let x = f64s(&random_f32(rows * d, 1));
    let w = f64s(&random_f32(d, 2));
    let dy = f64s(&random_f32(rows * d, 3));
    let (dx, dw) = rms_bwd(&x, &w, &dy, d, eps);
    let loss = |x: &[f64], w: &[f64]| rms_fwd(x, w, d, eps).iter().zip(&dy).map(|(a, b)| a * b).sum::<f64>();
    fd_check("dx", &x, &dx, &|v| loss(v, &w));
    fd_check("dw", &w, &dw, &|v| loss(&x, v));
}

fn run_rms(rt: &Arc<GpuRuntime>, rows: usize, d: usize, accumulate: bool, seed: u64) {
    let eps = 1e-6f32;
    let x = random_f32(rows * d, seed);
    let w: Vec<f32> = random_f32(d, seed + 1).iter().map(|v| 1.0 + 0.5 * v).collect();
    let dy = random_f32(rows * d, seed + 2);
    let prior = random_f32(rows * d, seed + 3);
    let (xb, wb, dyb) = (buf(rt, &x), buf(rt, &w), buf(rt, &dy));
    let dxb = if accumulate { buf(rt, &prior) } else { seeded(rt, rows * d, SENTINEL) };
    let dwb = seeded(rt, d, SENTINEL);
    let part = seeded(rt, rms_norm_bwd_part_len(rows as u32, d as u32).max(1), SENTINEL);
    rms_norm_bwd(rt, &xb, &wb, &dyb, &dxb, &dwb, &part, rows as u32, d as u32, eps, accumulate).unwrap();
    rt.synchronize().unwrap();
    let (mut want_dx, want_dw) = rms_bwd(&f64s(&x), &f64s(&w), &f64s(&dy), d, f64::from(eps));
    if accumulate {
        for (a, p) in want_dx.iter_mut().zip(&prior) {
            *a += f64::from(*p);
        }
    }
    let label = format!("rms rows={rows} d={d} acc={accumulate}");
    close(&format!("{label} dx"), &dxb.read_f32(), &want_dx);
    let first = dwb.read_f32();
    close(&format!("{label} dw"), &first, &want_dw);
    if !accumulate {
        rms_norm_bwd(rt, &xb, &wb, &dyb, &dxb, &dwb, &part, rows as u32, d as u32, eps, false).unwrap();
        rt.synchronize().unwrap();
        assert!(dwb.read_f32().iter().zip(&first).all(|(a, b)| a.to_bits() == b.to_bits()), "{label}: dw changed on a rerun");
    }
}

#[test]
fn rms_norm_backward_matches_across_blocks_and_widths() {
    with_gpu(|rt| {
        for (rows, d) in [(1, 64), (31, 256), (32, 2048), (33, 1000), (100, 2048), (65, 5120)] {
            run_rms(rt, rows, d, false, rows as u64 * 7 + d as u64);
        }
        run_rms(rt, 40, 2048, true, 99);
        // rows = 0 writes a zero weight gradient.
        let (xb, wb) = (buf(rt, &[0.0; 64]), buf(rt, &[1.0; 64]));
        let dwb = seeded(rt, 64, SENTINEL);
        let part = seeded(rt, 1, SENTINEL);
        let dyb = buf(rt, &[0.0; 64]);
        let dxb = seeded(rt, 64, 0.0);
        rms_norm_bwd(rt, &xb, &wb, &dyb, &dxb, &dwb, &part, 0, 64, 1e-6, false).unwrap();
        rt.synchronize().unwrap();
        assert!(dwb.read_f32().iter().all(|&v| v == 0.0), "rows = 0: dw must be zeros");
    });
}

// ------------------------------------------------------ gated RMSNorm ---

fn gated_fwd(x: &[f64], z: &[f64], w: &[f64], d: usize, eps: f64) -> Vec<f64> {
    x.chunks(d)
        .zip(z.chunks(d))
        .flat_map(|(xr, zr)| {
            let rstd = 1.0 / (xr.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
            (0..d).map(move |j| w[j] * xr[j] * rstd * silu(zr[j])).collect::<Vec<_>>()
        })
        .collect()
}

fn gated_bwd(x: &[f64], z: &[f64], w: &[f64], dy: &[f64], d: usize, eps: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut dx, mut dz, mut dw) = (vec![0.0; x.len()], vec![0.0; x.len()], vec![0.0; d]);
    for u in 0..x.len() / d {
        let (xr, zr, gr) = (&x[u * d..(u + 1) * d], &z[u * d..(u + 1) * d], &dy[u * d..(u + 1) * d]);
        let rstd = 1.0 / (xr.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
        let xn: Vec<f64> = xr.iter().map(|v| v * rstd).collect();
        let dxn: Vec<f64> = (0..d).map(|j| gr[j] * w[j] * silu(zr[j])).collect();
        let m: f64 = (0..d).map(|j| dxn[j] * xn[j]).sum::<f64>() / d as f64;
        for j in 0..d {
            dx[u * d + j] = rstd * (dxn[j] - xn[j] * m);
            dz[u * d + j] = gr[j] * w[j] * xn[j] * silu_grad(zr[j]);
            dw[j] += gr[j] * xn[j] * silu(zr[j]);
        }
    }
    (dx, dz, dw)
}

#[test]
fn gated_rms_norm_reference_backward_is_the_derivative() {
    let (units, d, eps) = (3, 6, 1e-6);
    let x = f64s(&random_f32(units * d, 11));
    let z = f64s(&random_f32(units * d, 12)).iter().map(|v| 3.0 * v).collect::<Vec<_>>();
    let w = f64s(&random_f32(d, 13));
    let dy = f64s(&random_f32(units * d, 14));
    let (dx, dz, dw) = gated_bwd(&x, &z, &w, &dy, d, eps);
    let loss = |x: &[f64], z: &[f64], w: &[f64]| gated_fwd(x, z, w, d, eps).iter().zip(&dy).map(|(a, b)| a * b).sum::<f64>();
    fd_check("dx", &x, &dx, &|v| loss(v, &z, &w));
    fd_check("dz", &z, &dz, &|v| loss(&x, v, &w));
    fd_check("dw", &w, &dw, &|v| loss(&x, &z, v));
}

/// Windows as the model lays them out: x the dense GDN core output, z a
/// window of the fused projection, dz written into that window of the fused
/// projection's gradient next to another window that must stay untouched.
fn run_gated(rt: &Arc<GpuRuntime>, rows: usize, heads: usize, d: usize, seed: u64) {
    let eps = 1e-6f32;
    let width = heads * d;
    let (ld_z, z_off) = (width + 48, 40);
    let x = random_f32(rows * width, seed);
    let zfull: Vec<f32> = random_f32(rows * ld_z, seed + 1).iter().map(|v| 4.0 * v).collect();
    let w = random_f32(d, seed + 2);
    let dy = random_f32(rows * width, seed + 3);
    let (xb, zb, wb, dyb) = (buf(rt, &x), buf(rt, &zfull), buf(rt, &w), buf(rt, &dy));
    let dxb = seeded(rt, rows * width, SENTINEL);
    let dzb = seeded(rt, rows * ld_z, SENTINEL);
    let dwb = seeded(rt, d, SENTINEL);
    let part = seeded(rt, gated_rms_norm_bwd_part_len(rows as u32, heads as u32, d as u32), SENTINEL);
    gated_rms_norm_bwd(rt, win(&xb, width, 0), win(&zb, ld_z, z_off), &wb, win(&dyb, width, 0), win(&dxb, width, 0), win(&dzb, ld_z, z_off), &dwb, &part, rows as u32, heads as u32, d as u32, eps).unwrap();
    rt.synchronize().unwrap();
    let z: Vec<f64> = (0..rows).flat_map(|r| f64s(&zfull[r * ld_z + z_off..r * ld_z + z_off + width])).collect();
    let (wdx, wdz, wdw) = gated_bwd(&f64s(&x), &z, &f64s(&w), &f64s(&dy), d, f64::from(eps));
    let label = format!("gated rows={rows} heads={heads} d={d}");
    close(&format!("{label} dx"), &dxb.read_f32(), &wdx);
    let dz_all = dzb.read_f32();
    let dz: Vec<f32> = (0..rows).flat_map(|r| dz_all[r * ld_z + z_off..r * ld_z + z_off + width].to_vec()).collect();
    close(&format!("{label} dz"), &dz, &wdz);
    for r in 0..rows {
        for c in (0..z_off).chain(z_off + width..ld_z) {
            assert_eq!(dz_all[r * ld_z + c], SENTINEL, "{label}: dz wrote outside its window at ({r}, {c})");
        }
    }
    close(&format!("{label} dw"), &dwb.read_f32(), &wdw);
}

#[test]
fn gated_rms_norm_backward_matches_in_the_model_layout() {
    with_gpu(|rt| {
        for (rows, heads, d) in [(1, 1, 128), (3, 16, 128), (5, 13, 128), (17, 4, 64), (2, 3, 512)] {
            run_gated(rt, rows, heads, d, (rows * heads * d) as u64);
        }
    });
}

// ------------------------------------------------ SwiGLU, output gate ---

#[test]
fn swiglu_and_gate_backward_match() {
    with_gpu(|rt| {
        // SwiGLU in the fused [T, 2I] layout: gate columns [0, I), up [I, 2I).
        let (rows, inter) = (37usize, 300usize);
        let ld = 2 * inter;
        let gu: Vec<f32> = random_f32(rows * ld, 21).iter().map(|v| 6.0 * v).collect();
        let dy = random_f32(rows * inter, 22);
        let (gub, dyb) = (buf(rt, &gu), buf(rt, &dy));
        let dgu = seeded(rt, rows * ld, SENTINEL);
        swiglu_bwd(rt, win(&gub, ld, 0), win(&gub, ld, inter), win(&dyb, inter, 0), win(&dgu, ld, 0), win(&dgu, ld, inter), rows as u32, inter as u32).unwrap();
        rt.synchronize().unwrap();
        let mut want = vec![0.0f64; rows * ld];
        for r in 0..rows {
            for c in 0..inter {
                let (g, u, d) = (f64::from(gu[r * ld + c]), f64::from(gu[r * ld + inter + c]), f64::from(dy[r * inter + c]));
                want[r * ld + c] = d * u * silu_grad(g);
                want[r * ld + inter + c] = d * silu(g);
            }
        }
        close("swiglu dgate|dup", &dgu.read_f32(), &want);

        // The output gate: p is the fused [rows, ld_p] projection, head h's
        // gate at q_off + h*2D + D; dp gets exactly those columns.
        let (rows, heads, d) = (9usize, 3usize, 32usize);
        let (ld_p, q_off) = (heads * 2 * d + 20, 8usize);
        let p: Vec<f32> = random_f32(rows * ld_p, 31).iter().map(|v| 5.0 * v).collect();
        let attn = random_f32(rows * heads * d, 32);
        let dy = random_f32(rows * heads * d, 33);
        let (pb, ab, dyb) = (buf(rt, &p), buf(rt, &attn), buf(rt, &dy));
        let (dab, dpb) = (seeded(rt, rows * heads * d, SENTINEL), seeded(rt, rows * ld_p, SENTINEL));
        attn_gate_bwd(rt, &ab, Cols { buf: &pb, ld: ld_p as u32, off: q_off as u32 }, Cols::dense(&dyb, (heads * d) as u32), &dab, &dpb, rows as u32, heads as u32, d as u32).unwrap();
        rt.synchronize().unwrap();
        let (mut wa, dp_all) = (vec![0.0f64; rows * heads * d], dpb.read_f32());
        let mut got_gate = Vec::new();
        let mut want_gate = Vec::new();
        for r in 0..rows {
            for h in 0..heads {
                for j in 0..d {
                    let gi = r * ld_p + q_off + h * 2 * d + d + j;
                    let s = sigmoid(f64::from(p[gi]));
                    let ai = r * heads * d + h * d + j;
                    wa[ai] = f64::from(dy[ai]) * s;
                    want_gate.push(f64::from(dy[ai]) * f64::from(attn[ai]) * s * (1.0 - s));
                    got_gate.push(dp_all[gi]);
                }
            }
        }
        close("gate d_attn", &dab.read_f32(), &wa);
        close("gate d_gate", &got_gate, &want_gate);
        let written = dp_all.iter().filter(|&&v| v != SENTINEL).count();
        assert_eq!(written, rows * heads * d, "attn_gate_bwd wrote outside the gate columns of dp");
    });
}

#[test]
fn the_backward_entry_points_refuse_bad_layouts() {
    with_gpu(|rt| {
        let a = buf(rt, &[0.0; 4096]);
        let b = buf(rt, &[0.0; 4096]);
        let c = buf(rt, &[0.0; 4096]);
        let e = |r: Result<(), String>, needle: &str| {
            let m = r.expect_err(needle);
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        };
        let part = buf(rt, &[0.0; 4096]);
        e(rms_norm_bwd(rt, &a, &b, &c, &a, &c, &part, 2, 64, 1e-6, false), "overlap");
        e(rms_norm_bwd(rt, &a, &b, &c, &part, &c, &part, 2, 64, 1e-6, false), "overlap");
        e(rms_norm_bwd(rt, &a, &b, &c, &buf(rt, &[0.0; 128]), &buf(rt, &[0.0; 64]), &buf(rt, &[0.0; 1]), 2, 64, 1e-6, false), "part");
        e(rms_norm_bwd(rt, &a, &b, &c, &buf(rt, &[0.0; 128]), &buf(rt, &[0.0; 64]), &part, 2, 64, 0.0, false), "eps");
        e(rms_norm_bwd(rt, &a, &b, &c, &buf(rt, &[0.0; 128]), &buf(rt, &[0.0; 64]), &part, 2, 64 * 1024 + 1, 1e-6, false), "rms_norm_bwd x");
        // SwiGLU: the two gradient windows may share a buffer only when disjoint.
        let w = |bf, off| win(bf, 64, off);
        e(swiglu_bwd(rt, w(&a, 0), w(&a, 32), w(&b, 0), w(&c, 0), w(&c, 16), 4, 32), "dgate overlaps dup");
        e(swiglu_bwd(rt, w(&a, 0), w(&a, 32), w(&b, 0), w(&a, 0), w(&c, 32), 4, 32), "dgate overlaps gate");
        e(swiglu_bwd(rt, w(&a, 0), w(&a, 40), w(&b, 0), w(&c, 0), w(&c, 32), 4, 32), "does not fit");
        swiglu_bwd(rt, w(&a, 0), w(&a, 32), w(&b, 0), w(&c, 0), w(&c, 32), 4, 32).expect("disjoint windows of one buffer");
        e(gated_rms_norm_bwd(rt, w(&a, 0), w(&a, 32), &b, w(&b, 0), w(&c, 0), w(&a, 32), &buf(rt, &[0.0; 32]), &part, 2, 1, 32, 1e-6), "dz overlaps z");
        e(gated_rms_norm_bwd(rt, w(&a, 0), w(&b, 0), &b, w(&b, 32), w(&c, 0), w(&c, 32), &buf(rt, &[0.0; 1]), &part, 2, 1, 600, 1e-6), "dim in 1..=512");
        e(attn_gate_bwd(rt, &a, Cols { buf: &b, ld: 64, off: 0 }, w(&c, 0), &a, &part, 2, 1, 32), "overlap");
    });
}
