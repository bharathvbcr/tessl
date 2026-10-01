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

use common::qwen35::{sigmoid, silu, softplus};
use common::{buf, random_f32, seeded, with_gpu};
use tessl::qwen35::{self, AttnShape, Cols, GdnGateLogits, GdnParams};
use tessl::qwen35_bwd::{
    attn_gate_bwd, attn_qk_norm_rope_bwd, attn_qk_norm_rope_bwd_part_len, conv1d_silu_bwd, conv1d_silu_bwd_part_len,
    copy_cols, embed_rows_bwd, gated_rms_norm_bwd, gated_rms_norm_bwd_part_len, gdn_gates_bwd, gdn_gates_bwd_part_len,
    rms_norm_bwd, rms_norm_bwd_part_len, scatter_add_rows, swiglu_bwd, AttnQkvGrads, EmbedBwdWorkspace,
};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

const SENTINEL: f32 = -7.25e27;

fn win(buf: &GpuBuffer, ld: usize, off: usize) -> Cols<'_> {
    Cols {
        buf,
        ld: ld as u32,
        off: off as u32,
    }
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
    assert!(
        rel <= 1e-4,
        "{label}: max err {:.3e} at {} (got {} want {}), {rel:.2e} of {peak:.3e}",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
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

/// `Qwen3_5RMSNorm`: `x * rstd * (1 + w)`, the zero-centred `w` as stored.
fn rms_fwd(x: &[f64], w: &[f64], d: usize, eps: f64) -> Vec<f64> {
    x.chunks(d)
        .flat_map(|r| {
            let rstd = 1.0 / (r.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
            r.iter()
                .zip(w)
                .map(move |(v, wv)| v * rstd * (1.0 + wv))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn rms_bwd(x: &[f64], w: &[f64], dy: &[f64], d: usize, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let mut dx = vec![0.0; x.len()];
    let mut dw = vec![0.0; d];
    for (r, (xr, gr)) in x.chunks(d).zip(dy.chunks(d)).enumerate() {
        let rstd = 1.0 / (xr.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
        let dot: f64 = (0..d).map(|j| gr[j] * (1.0 + w[j]) * xr[j]).sum();
        for j in 0..d {
            dx[r * d + j] = rstd * gr[j] * (1.0 + w[j]) - xr[j] * rstd.powi(3) * dot / d as f64;
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
    let w: Vec<f32> = random_f32(d, seed + 1).iter().map(|v| 0.5 * v).collect();
    let dy = random_f32(rows * d, seed + 2);
    let prior = random_f32(rows * d, seed + 3);
    let (xb, wb, dyb) = (buf(rt, &x), buf(rt, &w), buf(rt, &dy));
    let dxb = if accumulate {
        buf(rt, &prior)
    } else {
        seeded(rt, rows * d, SENTINEL)
    };
    let dwb = seeded(rt, d, SENTINEL);
    let part = seeded(rt, rms_norm_bwd_part_len(rows as u32, d as u32).max(1), SENTINEL);
    rms_norm_bwd(
        rt,
        &xb,
        &wb,
        &dyb,
        &dxb,
        &dwb,
        &part,
        rows as u32,
        d as u32,
        eps,
        accumulate,
    )
    .unwrap();
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
        assert!(
            dwb.read_f32()
                .iter()
                .zip(&first)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{label}: dw changed on a rerun"
        );
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
            (0..d)
                .map(move |j| w[j] * xr[j] * rstd * silu(zr[j]))
                .collect::<Vec<_>>()
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
    let z = f64s(&random_f32(units * d, 12))
        .iter()
        .map(|v| 3.0 * v)
        .collect::<Vec<_>>();
    let w = f64s(&random_f32(d, 13));
    let dy = f64s(&random_f32(units * d, 14));
    let (dx, dz, dw) = gated_bwd(&x, &z, &w, &dy, d, eps);
    let loss = |x: &[f64], z: &[f64], w: &[f64]| {
        gated_fwd(x, z, w, d, eps)
            .iter()
            .zip(&dy)
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };
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
    let part = seeded(
        rt,
        gated_rms_norm_bwd_part_len(rows as u32, heads as u32, d as u32),
        SENTINEL,
    );
    gated_rms_norm_bwd(
        rt,
        win(&xb, width, 0),
        win(&zb, ld_z, z_off),
        &wb,
        win(&dyb, width, 0),
        win(&dxb, width, 0),
        win(&dzb, ld_z, z_off),
        &dwb,
        &part,
        rows as u32,
        heads as u32,
        d as u32,
        eps,
    )
    .unwrap();
    rt.synchronize().unwrap();
    let z: Vec<f64> = (0..rows)
        .flat_map(|r| f64s(&zfull[r * ld_z + z_off..r * ld_z + z_off + width]))
        .collect();
    let (wdx, wdz, wdw) = gated_bwd(&f64s(&x), &z, &f64s(&w), &f64s(&dy), d, f64::from(eps));
    let label = format!("gated rows={rows} heads={heads} d={d}");
    close(&format!("{label} dx"), &dxb.read_f32(), &wdx);
    let dz_all = dzb.read_f32();
    let dz: Vec<f32> = (0..rows)
        .flat_map(|r| dz_all[r * ld_z + z_off..r * ld_z + z_off + width].to_vec())
        .collect();
    close(&format!("{label} dz"), &dz, &wdz);
    for r in 0..rows {
        for c in (0..z_off).chain(z_off + width..ld_z) {
            assert_eq!(
                dz_all[r * ld_z + c],
                SENTINEL,
                "{label}: dz wrote outside its window at ({r}, {c})"
            );
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
        swiglu_bwd(
            rt,
            win(&gub, ld, 0),
            win(&gub, ld, inter),
            win(&dyb, inter, 0),
            win(&dgu, ld, 0),
            win(&dgu, ld, inter),
            rows as u32,
            inter as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let mut want = vec![0.0f64; rows * ld];
        for r in 0..rows {
            for c in 0..inter {
                let (g, u, d) = (
                    f64::from(gu[r * ld + c]),
                    f64::from(gu[r * ld + inter + c]),
                    f64::from(dy[r * inter + c]),
                );
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
        let (dab, dpb) = (
            seeded(rt, rows * heads * d, SENTINEL),
            seeded(rt, rows * ld_p, SENTINEL),
        );
        attn_gate_bwd(
            rt,
            &ab,
            Cols {
                buf: &pb,
                ld: ld_p as u32,
                off: q_off as u32,
            },
            Cols::dense(&dyb, (heads * d) as u32),
            &dab,
            &dpb,
            rows as u32,
            heads as u32,
            d as u32,
        )
        .unwrap();
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
        assert_eq!(
            written,
            rows * heads * d,
            "attn_gate_bwd wrote outside the gate columns of dp"
        );
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
        e(
            rms_norm_bwd(rt, &a, &b, &c, &a, &c, &part, 2, 64, 1e-6, false),
            "overlap",
        );
        e(
            rms_norm_bwd(rt, &a, &b, &c, &part, &c, &part, 2, 64, 1e-6, false),
            "overlap",
        );
        e(
            rms_norm_bwd(
                rt,
                &a,
                &b,
                &c,
                &buf(rt, &[0.0; 128]),
                &buf(rt, &[0.0; 64]),
                &buf(rt, &[0.0; 1]),
                2,
                64,
                1e-6,
                false,
            ),
            "part",
        );
        e(
            rms_norm_bwd(
                rt,
                &a,
                &b,
                &c,
                &buf(rt, &[0.0; 128]),
                &buf(rt, &[0.0; 64]),
                &part,
                2,
                64,
                0.0,
                false,
            ),
            "eps",
        );
        e(
            rms_norm_bwd(
                rt,
                &a,
                &b,
                &c,
                &buf(rt, &[0.0; 128]),
                &buf(rt, &[0.0; 64]),
                &part,
                2,
                64 * 1024 + 1,
                1e-6,
                false,
            ),
            "rms_norm_bwd x",
        );
        // SwiGLU: the two gradient windows may share a buffer only when disjoint.
        let w = |bf, off| win(bf, 64, off);
        e(
            swiglu_bwd(rt, w(&a, 0), w(&a, 32), w(&b, 0), w(&c, 0), w(&c, 16), 4, 32),
            "dgate overlaps dup",
        );
        e(
            swiglu_bwd(rt, w(&a, 0), w(&a, 32), w(&b, 0), w(&a, 0), w(&c, 32), 4, 32),
            "dgate overlaps gate",
        );
        e(
            swiglu_bwd(rt, w(&a, 0), w(&a, 40), w(&b, 0), w(&c, 0), w(&c, 32), 4, 32),
            "does not fit",
        );
        swiglu_bwd(rt, w(&a, 0), w(&a, 32), w(&b, 0), w(&c, 0), w(&c, 32), 4, 32)
            .expect("disjoint windows of one buffer");
        e(
            gated_rms_norm_bwd(
                rt,
                w(&a, 0),
                w(&a, 32),
                &b,
                w(&b, 0),
                w(&c, 0),
                w(&a, 32),
                &buf(rt, &[0.0; 32]),
                &part,
                2,
                1,
                32,
                1e-6,
            ),
            "dz overlaps z",
        );
        e(
            gated_rms_norm_bwd(
                rt,
                w(&a, 0),
                w(&b, 0),
                &b,
                w(&b, 32),
                w(&c, 0),
                w(&c, 32),
                &buf(rt, &[0.0; 1]),
                &part,
                2,
                1,
                600,
                1e-6,
            ),
            "dim in 1..=512",
        );
        e(
            attn_gate_bwd(
                rt,
                &a,
                Cols {
                    buf: &b,
                    ld: 64,
                    off: 0,
                },
                w(&c, 0),
                &a,
                &part,
                2,
                1,
                32,
            ),
            "overlap",
        );
    });
}

// ------------------------------------------------------ causal conv + SiLU ---

/// `y [B, T, C]` of the zero-state depthwise causal conv + SiLU, in f64.
fn conv_fwd(x: &[f64], w: &[f64], b: usize, t: usize, c: usize, kw: usize) -> Vec<f64> {
    let hist = kw - 1;
    let mut y = vec![0.0; b * t * c];
    for bi in 0..b {
        for ti in 0..t {
            for ci in 0..c {
                let pre: f64 = (0..kw)
                    .filter(|&j| ti + j >= hist)
                    .map(|j| w[ci * kw + j] * x[(bi * t + ti + j - hist) * c + ci])
                    .sum();
                y[(bi * t + ti) * c + ci] = silu(pre);
            }
        }
    }
    y
}

/// `(dx [B, T, C], dw [C, KW])` of [`conv_fwd`] for the upstream `dy`, by
/// scattering each output's `dpre` onto the inputs and taps it read.
#[allow(clippy::too_many_arguments)]
fn conv_bwd(x: &[f64], w: &[f64], dy: &[f64], b: usize, t: usize, c: usize, kw: usize) -> (Vec<f64>, Vec<f64>) {
    let hist = kw - 1;
    let (mut dx, mut dw) = (vec![0.0; b * t * c], vec![0.0; c * kw]);
    for bi in 0..b {
        for ti in 0..t {
            for ci in 0..c {
                let taps: Vec<usize> = (0..kw).filter(|&j| ti + j >= hist).collect();
                let at = |j: usize| (bi * t + ti + j - hist) * c + ci;
                let pre: f64 = taps.iter().map(|&j| w[ci * kw + j] * x[at(j)]).sum();
                let dpre = dy[(bi * t + ti) * c + ci] * silu_grad(pre);
                for &j in &taps {
                    dx[at(j)] += dpre * w[ci * kw + j];
                    dw[ci * kw + j] += dpre * x[at(j)];
                }
            }
        }
    }
    (dx, dw)
}

#[test]
fn conv_reference_backward_is_the_derivative() {
    // T = 2 < KW - 1 with B = 2: a batch row's gradient must not reach into
    // its neighbour's tokens.
    for (b, t, c, kw) in [(2, 2, 2, 4), (2, 5, 3, 4), (1, 4, 2, 2), (2, 3, 1, 8)] {
        let x = f64s(&random_f32(b * t * c, 41));
        let w = f64s(&random_f32(c * kw, 42));
        let dy = f64s(&random_f32(b * t * c, 43));
        let (dx, dw) = conv_bwd(&x, &w, &dy, b, t, c, kw);
        let loss = |x: &[f64], w: &[f64]| {
            conv_fwd(x, w, b, t, c, kw)
                .iter()
                .zip(&dy)
                .map(|(a, g)| a * g)
                .sum::<f64>()
        };
        fd_check("conv dx", &x, &dx, &|v| loss(v, &w));
        fd_check("conv dw", &w, &dw, &|v| loss(&x, v));
    }
    // The forward here is the zero-state forward the inference kernels are
    // checked against.
    let (b, t, c, kw) = (2, 6, 3, 4);
    let (x, w) = (random_f32(b * t * c, 44), random_f32(c * kw, 45));
    let (want, _) = common::qwen35::conv1d_silu_f64(&x, &w, None, b, t, c, kw);
    let got = conv_fwd(&f64s(&x), &f64s(&w), b, t, c, kw);
    assert!(
        got.iter().zip(&want).all(|(a, b)| (a - b).abs() <= 1e-12),
        "conv_fwd is not the tested forward"
    );
}

fn run_conv(rt: &Arc<GpuRuntime>, b: usize, t: usize, c: usize, kw: usize, seed: u64) {
    // x and dy as windows of wider rows, dx as a window of a fused gradient;
    // every column outside dx's window must be left alone.
    let rows = b * t;
    let (ld_x, x_off, ld_dy, dy_off, ld_dx, dx_off) = (c + 7, 3, c + 2, 2, c + 5, 1);
    let xs: Vec<f32> = random_f32(rows * ld_x, seed).iter().map(|v| 2.0 * v).collect();
    let w = random_f32(c * kw, seed + 1);
    let dys = random_f32(rows * ld_dy, seed + 2);
    let (xb, wb, dyb) = (buf(rt, &xs), buf(rt, &w), buf(rt, &dys));
    let dxb = seeded(rt, rows * ld_dx, SENTINEL);
    let dwb = seeded(rt, c * kw, SENTINEL);
    let part = seeded(
        rt,
        conv1d_silu_bwd_part_len(b as u32, t as u32, c as u32, kw as u32).max(1),
        SENTINEL,
    );
    let call = || {
        conv1d_silu_bwd(
            rt,
            win(&xb, ld_x, x_off),
            &wb,
            kw as u32,
            win(&dyb, ld_dy, dy_off),
            win(&dxb, ld_dx, dx_off),
            &dwb,
            &part,
            b as u32,
            t as u32,
            c as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();
    };
    call();
    let dense = |v: &[f32], ld: usize, off: usize| -> Vec<f64> {
        (0..rows)
            .flat_map(|r| (0..c).map(move |j| (r, j)))
            .map(|(r, j)| f64::from(v[r * ld + off + j]))
            .collect()
    };
    let (want_dx, want_dw) = conv_bwd(
        &dense(&xs, ld_x, x_off),
        &f64s(&w),
        &dense(&dys, ld_dy, dy_off),
        b,
        t,
        c,
        kw,
    );
    let label = format!("conv b={b} t={t} c={c} kw={kw}");
    let dx_all = dxb.read_f32();
    let got_dx: Vec<f32> = (0..rows)
        .flat_map(|r| (0..c).map(move |j| (r, j)))
        .map(|(r, j)| dx_all[r * ld_dx + dx_off + j])
        .collect();
    close(&format!("{label} dx"), &got_dx, &want_dx);
    let written = dx_all.iter().filter(|&&v| v != SENTINEL).count();
    assert_eq!(written, rows * c, "{label}: wrote outside dx's window");
    let first = dwb.read_f32();
    close(&format!("{label} dw"), &first, &want_dw);
    call();
    assert!(
        dwb.read_f32()
            .iter()
            .zip(&first)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: dw changed on a rerun"
    );
}

#[test]
fn conv_backward_matches_across_batch_rows_blocks_and_widths() {
    with_gpu(|rt| {
        for (b, t, c, kw) in [
            (2, 1, 5, 4),   // every token is inside the zero padding
            (3, 2, 7, 4),   // T < KW - 1 across batch rows
            (2, 9, 6, 8),   // widest kernel
            (1, 5, 4, 2),   // narrowest
            (1, 256, 8, 4), // exactly one weight-gradient block
            (1, 257, 8, 4), // one row into the next block
            (3, 200, 33, 4),
            (2, 64, 6144, 4), // the 2B's conv width
        ] {
            run_conv(rt, b, t, c, kw, (b * 1000 + t * 10 + c) as u64);
        }
        // No rows: a zero weight gradient, no input gradient written.
        let (xb, wb) = (buf(rt, &[0.0; 16]), buf(rt, &[1.0; 16]));
        let dwb = seeded(rt, 16, SENTINEL);
        let (dyb, dxb) = (buf(rt, &[0.0; 16]), seeded(rt, 16, SENTINEL));
        conv1d_silu_bwd(
            rt,
            Cols::dense(&xb, 4),
            &wb,
            4,
            Cols::dense(&dyb, 4),
            Cols::dense(&dxb, 4),
            &dwb,
            &buf(rt, &[0.0; 1]),
            2,
            0,
            4,
        )
        .unwrap();
        rt.synchronize().unwrap();
        assert!(dwb.read_f32().iter().all(|&v| v == 0.0), "seq = 0: dw must be zeros");
        assert!(
            dxb.read_f32().iter().all(|&v| v == SENTINEL),
            "seq = 0: dx must be untouched"
        );
    });
}

#[test]
fn conv_backward_refuses_bad_layouts() {
    with_gpu(|rt| {
        let (a, b, c) = (buf(rt, &[0.0; 4096]), buf(rt, &[0.0; 4096]), buf(rt, &[0.0; 4096]));
        let (wb, dwb, part) = (buf(rt, &[0.0; 256]), buf(rt, &[0.0; 256]), buf(rt, &[0.0; 4096]));
        let e = |r: Result<(), String>, needle: &str| {
            let m = r.expect_err(needle);
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        };
        let w = |bf, off| win(bf, 64, off);
        let run = |x, kw, dy, dx, dw: &GpuBuffer, part: &GpuBuffer, seq| {
            conv1d_silu_bwd(rt, x, &wb, kw, dy, dx, dw, part, 2, seq, 32)
        };
        run(w(&a, 0), 4, w(&b, 0), w(&a, 32), &dwb, &part, 8).expect("dx beside x in one buffer");
        e(
            run(w(&a, 0), 1, w(&b, 0), w(&c, 0), &dwb, &part, 8),
            "kernel_width must be 2..=8",
        );
        e(
            run(w(&a, 0), 9, w(&b, 0), w(&c, 0), &dwb, &part, 8),
            "kernel_width must be 2..=8",
        );
        e(run(w(&a, 0), 4, w(&b, 0), w(&a, 16), &dwb, &part, 8), "dx overlaps x");
        e(run(w(&a, 0), 4, w(&b, 0), w(&b, 8), &dwb, &part, 8), "dx overlaps dy");
        e(
            run(w(&a, 0), 4, w(&b, 0), w(&c, 0), &dwb, &buf(rt, &[0.0; 1]), 8),
            "part",
        );
        e(run(w(&a, 0), 4, w(&b, 0), w(&c, 0), &a, &part, 8), "dw");
        e(
            run(w(&a, 0), 4, w(&b, 0), w(&c, 0), &dwb, &part, 40),
            "conv1d_silu_bwd x: buffer holds",
        );
        e(
            run(w(&a, 0), 4, w(&b, 0), w(&c, 0), &buf(rt, &[0.0; 8]), &part, 8),
            "dw",
        );
    });
}

// ------------------------------------------------- Q/K norm + partial RoPE ---

/// cos and sin of transformers' angle for pair `p`, the angle formed in f32
/// as the model forms it (see `common::qwen35::norm_rope_row_f64`).
fn rope_cs(p: usize, rot: usize, pos: usize, theta: f64) -> (f64, f64) {
    let inv_freq = 1.0f32 / (theta as f32).powf((2 * p) as f32 / rot as f32);
    let angle = f64::from(pos as f32 * inv_freq);
    (angle.cos(), angle.sin())
}

/// One head row through the `(1 + w)` norm and partial RoPE, in f64.
fn norm_rope_fwd(x: &[f64], w: &[f64], rot: usize, pos: usize, theta: f64, eps: f64) -> Vec<f64> {
    let d = x.len();
    let rstd = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
    let mut n: Vec<f64> = x.iter().zip(w).map(|(v, wi)| v * rstd * (1.0 + wi)).collect();
    let half = rot / 2;
    for p in 0..half {
        let (c, s) = rope_cs(p, rot, pos, theta);
        let (n0, n1) = (n[p], n[p + half]);
        n[p] = n0 * c - n1 * s;
        n[p + half] = n1 * c + n0 * s;
    }
    n
}

/// `(dx, dw)` of [`norm_rope_fwd`] for the output gradient `g`.
fn norm_rope_bwd(
    x: &[f64],
    w: &[f64],
    g: &[f64],
    rot: usize,
    pos: usize,
    theta: f64,
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    let d = x.len();
    let rstd = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
    let mut dn = g.to_vec();
    let half = rot / 2;
    for p in 0..half {
        let (c, s) = rope_cs(p, rot, pos, theta);
        dn[p] = g[p] * c + g[p + half] * s;
        dn[p + half] = g[p + half] * c - g[p] * s;
    }
    let xn: Vec<f64> = x.iter().map(|v| v * rstd).collect();
    let dxn: Vec<f64> = dn.iter().zip(w).map(|(a, wi)| a * (1.0 + wi)).collect();
    let m = dxn.iter().zip(&xn).map(|(a, b)| a * b).sum::<f64>() / d as f64;
    let dx = dxn.iter().zip(&xn).map(|(a, b)| rstd * (a - b * m)).collect();
    let dw = dn.iter().zip(&xn).map(|(a, b)| a * b).collect();
    (dx, dw)
}

#[test]
fn qk_norm_rope_reference_backward_is_the_derivative() {
    for (d, rot, pos) in [(8, 8, 3), (10, 4, 7), (6, 0, 1), (12, 6, 40)] {
        let x = f64s(&random_f32(d, 51));
        let w = f64s(&random_f32(d, 52));
        let g = f64s(&random_f32(d, 53));
        let (dx, dw) = norm_rope_bwd(&x, &w, &g, rot, pos, 1e4, 1e-6);
        let loss = |x: &[f64], w: &[f64]| {
            norm_rope_fwd(x, w, rot, pos, 1e4, 1e-6)
                .iter()
                .zip(&g)
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        fd_check("qk dx", &x, &dx, &|v| loss(v, &w));
        fd_check("qk dw", &w, &dw, &|v| loss(&x, v));
    }
    // The forward here is the transformers-anchored one the forward kernel
    // is checked against.
    let (x, w) = (random_f32(256, 54), random_f32(256, 55));
    let want = common::qwen35::norm_rope_row_f64(&x, &w, 64, 1234, 1e7, 1e-6);
    let got = norm_rope_fwd(&f64s(&x), &f64s(&w), 64, 1234, 1e7, 1e-6);
    assert!(
        got.iter().zip(&want).all(|(a, b)| (a - b).abs() <= 1e-12),
        "norm_rope_fwd is not the tested forward"
    );
}

#[allow(clippy::too_many_arguments)]
fn run_qk(
    rt: &Arc<GpuRuntime>,
    b: usize,
    t: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    rot: usize,
    theta: f32,
    seed: u64,
) {
    let eps = 1e-6f32;
    let rows = b * t;
    let width = 2 * (hq + hkv) * d;
    let (ld, off) = (width + 5, 3usize);
    let shape = AttnShape {
        batch: b as u32,
        seq: t as u32,
        q_heads: hq as u32,
        kv_heads: hkv as u32,
        head_dim: d as u32,
        rotary_dim: rot as u32,
    };
    let p: Vec<f32> = random_f32(rows * ld, seed).iter().map(|v| 3.0 * v).collect();
    let (qw, kw) = (random_f32(d, seed + 1), random_f32(d, seed + 2));
    let (dq, dk, dv) = (
        random_f32(rows * hq * d, seed + 3),
        random_f32(rows * hkv * d, seed + 4),
        random_f32(rows * hkv * d, seed + 5),
    );
    let (pb, qwb, kwb) = (buf(rt, &p), buf(rt, &qw), buf(rt, &kw));
    let (dqb, dkb, dvb) = (buf(rt, &dq), buf(rt, &dk), buf(rt, &dv));
    let dpb = seeded(rt, rows * ld, SENTINEL);
    let (dqw, dkw) = (seeded(rt, d, SENTINEL), seeded(rt, d, SENTINEL));
    let part = seeded(rt, attn_qk_norm_rope_bwd_part_len(&shape).max(1), SENTINEL);
    let grads = AttnQkvGrads {
        dq: &dqb,
        dk: &dkb,
        dv: &dvb,
    };
    let call = || {
        attn_qk_norm_rope_bwd(
            rt,
            &shape,
            win(&pb, ld, off),
            &qwb,
            &kwb,
            &grads,
            &dpb,
            &dqw,
            &dkw,
            &part,
            theta,
            eps,
        )
        .unwrap();
        rt.synchronize().unwrap();
    };
    call();
    let (qw64, kw64) = (f64s(&qw), f64s(&kw));
    let (mut want_dp, mut got_dp) = (Vec::new(), Vec::new());
    let (mut want_dqw, mut want_dkw) = (vec![0.0f64; d], vec![0.0f64; d]);
    let dp = dpb.read_f32();
    let mut touched = 0usize;
    for r in 0..rows {
        let pos = r % t;
        let base = r * ld + off;
        let mut head = |col: usize, w: &[f64], g: &[f32], acc: &mut Vec<f64>| {
            let x = f64s(&p[base + col..base + col + d]);
            let (dx, dw) = norm_rope_bwd(&x, w, &f64s(g), rot, pos, f64::from(theta), f64::from(eps));
            want_dp.extend(dx);
            got_dp.extend_from_slice(&dp[base + col..base + col + d]);
            acc.iter_mut().zip(dw).for_each(|(a, v)| *a += v);
        };
        for h in 0..hq {
            head(h * 2 * d, &qw64, &dq[(r * hq + h) * d..][..d], &mut want_dqw);
        }
        for h in 0..hkv {
            head(2 * hq * d + h * d, &kw64, &dk[(r * hkv + h) * d..][..d], &mut want_dkw);
        }
        for h in 0..hkv {
            let col = base + 2 * hq * d + hkv * d + h * d;
            want_dp.extend(f64s(&dv[(r * hkv + h) * d..][..d]));
            got_dp.extend_from_slice(&dp[col..col + d]);
        }
        touched += (hq + 2 * hkv) * d;
    }
    let label = format!("qk b={b} t={t} hq={hq} hkv={hkv} d={d} rot={rot}");
    close(&format!("{label} dproj"), &got_dp, &want_dp);
    assert_eq!(
        dp.iter().filter(|&&v| v != SENTINEL).count(),
        touched,
        "{label}: wrote outside the q, k, v columns"
    );
    let (first_q, first_k) = (dqw.read_f32(), dkw.read_f32());
    close(&format!("{label} dq_norm_w"), &first_q, &want_dqw);
    close(&format!("{label} dk_norm_w"), &first_k, &want_dkw);
    call();
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        same(&dqw.read_f32(), &first_q) && same(&dkw.read_f32(), &first_k),
        "{label}: norm weight gradients changed on a rerun"
    );
}

#[test]
fn qk_norm_rope_backward_matches_across_blocks_and_layouts() {
    with_gpu(|rt| {
        run_qk(rt, 1, 1, 1, 1, 64, 64, 1e4, 1); // fully rotary, position 0
        run_qk(rt, 3, 7, 2, 1, 32, 8, 1e4, 2); // positions restart per batch row
        run_qk(rt, 1, 16, 2, 2, 64, 16, 1e4, 3); // exactly one block
        run_qk(rt, 2, 17, 4, 2, 128, 32, 1e6, 4); // blocks straddle batch rows
        run_qk(rt, 1, 20, 3, 1, 96, 24, 1e4, 5); // pairs split across lanes
        run_qk(rt, 1, 5, 2, 1, 64, 0, 1e4, 6); // no rotary dims
        run_qk(rt, 1, 300, 8, 2, 256, 64, 1e7, 7); // the 2B's heads
                                                   // No tokens: zero norm-weight gradients.
        let shape = AttnShape {
            batch: 2,
            seq: 0,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 32,
            rotary_dim: 8,
        };
        let z = buf(rt, &[0.0; 256]);
        let (dqw, dkw) = (seeded(rt, 32, SENTINEL), seeded(rt, 32, SENTINEL));
        let grads = AttnQkvGrads { dq: &z, dk: &z, dv: &z };
        let dp = seeded(rt, 256, SENTINEL);
        attn_qk_norm_rope_bwd(
            rt,
            &shape,
            Cols::dense(&z, 192),
            &z,
            &z,
            &grads,
            &dp,
            &dqw,
            &dkw,
            &buf(rt, &[0.0; 1]),
            1e4,
            1e-6,
        )
        .unwrap();
        rt.synchronize().unwrap();
        assert!(
            dqw.read_f32().iter().chain(&dkw.read_f32()).all(|&v| v == 0.0),
            "seq = 0: norm gradients must be zeros"
        );
    });
}

#[test]
fn qk_norm_rope_backward_refuses_bad_layouts() {
    with_gpu(|rt| {
        let e = |r: Result<(), String>, needle: &str| {
            let m = r.expect_err(needle);
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        };
        let shape = AttnShape {
            batch: 1,
            seq: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 32,
            rotary_dim: 8,
        };
        let (p, g, w) = (buf(rt, &[0.0; 1024]), buf(rt, &[0.0; 512]), buf(rt, &[0.0; 32]));
        let (dp, dqw, dkw, part) = (
            buf(rt, &[0.0; 1024]),
            buf(rt, &[0.0; 32]),
            buf(rt, &[0.0; 32]),
            buf(rt, &[0.0; 64]),
        );
        let grads = AttnQkvGrads { dq: &g, dk: &g, dv: &g };
        let run = |s: &AttnShape, dp: &GpuBuffer, part: &GpuBuffer, theta: f32| {
            attn_qk_norm_rope_bwd(
                rt,
                s,
                Cols::dense(&p, 192),
                &w,
                &w,
                &grads,
                dp,
                &dqw,
                &dkw,
                part,
                theta,
                1e-6,
            )
        };
        run(&shape, &dp, &part, 1e4).expect("a valid call");
        e(
            run(&AttnShape { rotary_dim: 7, ..shape }, &dp, &part, 1e4),
            "rotary_dim must be even",
        );
        e(
            run(
                &AttnShape {
                    rotary_dim: 34,
                    ..shape
                },
                &dp,
                &part,
                1e4,
            ),
            "rotary_dim must be even and at most head_dim",
        );
        e(
            run(
                &AttnShape {
                    head_dim: 544,
                    rotary_dim: 8,
                    ..shape
                },
                &dp,
                &part,
                1e4,
            ),
            "head_dim must be at most 512",
        );
        e(run(&shape, &dp, &part, 0.0), "theta and eps");
        e(run(&shape, &p, &part, 1e4), "dproj");
        e(run(&shape, &dp, &buf(rt, &[0.0; 63]), 1e4), "part");
        e(
            run(&AttnShape { seq: 6, ..shape }, &dp, &buf(rt, &[0.0; 64]), 1e4),
            "proj",
        );
        e(run(&shape, &dp, &dqw, 1e4), "part");
    });
}

// ------------------------------------------------------ embedding gather ---

fn run_embed(rt: &Arc<GpuRuntime>, ws: &EmbedBwdWorkspace, ids: &[u32], vocab: usize, hidden: usize, seed: u64) {
    let rows = ids.len();
    let dh = random_f32(rows * hidden, seed);
    let prior = random_f32(vocab * hidden, seed + 1);
    let (dhb, dwb) = (buf(rt, &dh), buf(rt, &prior));
    embed_rows_bwd(rt, ids, &dhb, &dwb, vocab as u32, hidden as u32, ws).unwrap();
    rt.synchronize().unwrap();
    let mut want = f64s(&prior);
    for (r, &id) in ids.iter().enumerate() {
        for c in 0..hidden {
            want[id as usize * hidden + c] += f64::from(dh[r * hidden + c]);
        }
    }
    let got = dwb.read_f32();
    let label = format!("embed rows={rows} vocab={vocab} hidden={hidden}");
    close(&label, &got, &want);
    // Rows no id reads keep their bits.
    for v in 0..vocab {
        if !ids.contains(&(v as u32)) {
            let row = v * hidden..(v + 1) * hidden;
            assert!(
                got[row.clone()]
                    .iter()
                    .zip(&prior[row])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{label}: row {v} changed"
            );
        }
    }
    // The same call from the same prior gives the same bits.
    let again = buf(rt, &prior);
    embed_rows_bwd(rt, ids, &dhb, &again, vocab as u32, hidden as u32, ws).unwrap();
    rt.synchronize().unwrap();
    assert!(
        again
            .read_f32()
            .iter()
            .zip(&got)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: changed on a rerun"
    );
}

#[test]
fn embedding_backward_adds_each_ids_rows() {
    with_gpu(|rt| {
        let ws = EmbedBwdWorkspace::new(rt, 512).unwrap();
        run_embed(rt, &ws, &[3], 8, 64, 1);
        run_embed(rt, &ws, &[5, 0, 5, 7, 0, 5, 2, 7, 7, 1], 8, 100, 2); // repeats, out of order, ends of the table
        run_embed(rt, &ws, &[4; 300], 9, 33, 3); // one id, one long run
        let distinct: Vec<u32> = (0..257).rev().collect();
        run_embed(rt, &ws, &distinct, 300, 16, 4);
        let mixed: Vec<u32> = (0..512u32).map(|r| (r * 37) % 100 + 900).collect(); // ~5 rows per id
        run_embed(rt, &ws, &mixed, 1000, 2048, 5); // the 2B's hidden size
                                                   // A shorter call after a longer one: the stale tail of the workspace
                                                   // must not be read.
        run_embed(rt, &ws, &[1, 1, 6], 8, 40, 6);
        // No rows: dw is untouched.
        let dwb = buf(rt, &[2.5; 64]);
        embed_rows_bwd(rt, &[], &buf(rt, &[0.0; 1]), &dwb, 4, 16, &ws).unwrap();
        rt.synchronize().unwrap();
        assert!(
            dwb.read_f32().iter().all(|&v| v == 2.5),
            "no rows: dw must be untouched"
        );
    });
}

#[test]
fn embedding_backward_refuses_bad_ids_and_layouts() {
    with_gpu(|rt| {
        let e = |r: Result<(), String>, needle: &str| {
            let m = r.expect_err(needle);
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        };
        assert!(EmbedBwdWorkspace::new(rt, 0).is_err(), "max_rows 0");
        let ws = EmbedBwdWorkspace::new(rt, 4).unwrap();
        let (dh, dw) = (buf(rt, &[0.0; 64]), buf(rt, &[0.0; 128]));
        embed_rows_bwd(rt, &[0, 7, 3], &dh, &dw, 8, 16, &ws).expect("a valid call");
        e(
            embed_rows_bwd(rt, &[0, 8, 3], &dh, &dw, 8, 16, &ws),
            "ids[1] = 8 is not below vocab 8",
        );
        e(
            embed_rows_bwd(rt, &[0; 5], &dh, &dw, 8, 16, &ws),
            "5 rows exceed the workspace's 4",
        );
        e(embed_rows_bwd(rt, &[0; 4], &buf(rt, &[0.0; 63]), &dw, 8, 16, &ws), "dh");
        e(
            embed_rows_bwd(rt, &[0; 2], &dh, &buf(rt, &[0.0; 127]), 8, 16, &ws),
            "dw",
        );
        e(embed_rows_bwd(rt, &[0; 2], &dw, &dw, 8, 16, &ws), "dw");
    });
}

// --------------------------------------------------------------- GDN gates ---

/// torch's softplus derivative: sigmoid below the threshold, 1 above it.
fn softplus_grad(x: f64) -> f64 {
    if x > 20.0 {
        1.0
    } else {
        sigmoid(x)
    }
}

#[test]
fn gate_reference_backward_is_the_derivative() {
    // Away from the threshold kink, where central differences are exact.
    let (a_log, dt) = (0.3f64, -2.5f64);
    for a in [-18.0f64, -4.0, -0.5, 0.0, 2.0, 15.0, 30.0] {
        let g = |a: f64, al: f64, dt: f64| -al.exp() * softplus(a + dt);
        let h = 1e-6;
        let fd = |f: &dyn Fn(f64) -> f64, x: f64| (f(x + h) - f(x - h)) / (2.0 * h);
        let da = -a_log.exp() * softplus_grad(a + dt);
        let want = [
            (fd(&|x| g(x, a_log, dt), a), da),
            (fd(&|x| g(a, x, dt), a_log), g(a, a_log, dt)),
            (fd(&|x| g(a, a_log, x), dt), da),
        ];
        for (i, (f, got)) in want.iter().enumerate() {
            assert!(
                (f - got).abs() <= 1e-7 * (1.0 + f.abs()),
                "a = {a}, derivative {i}: {got} vs {f}"
            );
        }
    }
}

fn run_gates(rt: &Arc<GpuRuntime>, rows: usize, heads: usize, seed: u64) {
    // The fused GDN projection row: [other | b (H) | a (H) | other].
    let (b_off, a_off) = (5usize, 5 + heads);
    let ld = a_off + heads + 3;
    let mut p: Vec<f32> = random_f32(rows * ld, seed).iter().map(|v| 12.0 * v).collect();
    // Past torch's threshold (x = a + dt_bias > 20) and deep in the series.
    for r in (0..rows).step_by(3) {
        p[r * ld + a_off] = 26.0;
    }
    for r in (1..rows).step_by(5) {
        p[r * ld + a_off + heads - 1] = -24.0;
    }
    // Where softplus must not form e^x at all (it overflows f32 past 88).
    p[(rows - 1) * ld + a_off] = 120.0;
    let a_log: Vec<f32> = random_f32(heads, seed + 1).iter().map(|v| 0.8 * v).collect();
    let dt: Vec<f32> = random_f32(heads, seed + 2).iter().map(|v| -3.0 + 2.0 * v).collect();
    let (dg, dbeta) = (random_f32(rows * heads, seed + 3), random_f32(rows * heads, seed + 4));
    let (pb, alb, dtb) = (buf(rt, &p), buf(rt, &a_log), buf(rt, &dt));
    let (dgb, dbb) = (buf(rt, &dg), buf(rt, &dbeta));
    let logits = GdnGateLogits {
        buf: &pb,
        ld: ld as u32,
        a_off: a_off as u32,
        b_off: b_off as u32,
    };
    let params = GdnParams {
        a_log: &alb,
        dt_bias: &dtb,
    };
    let (gb, betab) = (seeded(rt, rows * heads, SENTINEL), seeded(rt, rows * heads, SENTINEL));
    qwen35::gdn_gates(rt, &logits, &params, &gb, &betab, rows as u32, heads as u32).unwrap();
    let dpb = seeded(rt, rows * ld, SENTINEL);
    let (dal, ddt) = (seeded(rt, heads, SENTINEL), seeded(rt, heads, SENTINEL));
    let part = seeded(rt, gdn_gates_bwd_part_len(rows as u32, heads as u32).max(1), SENTINEL);
    let bwd = || {
        gdn_gates_bwd(
            rt,
            &logits,
            &params,
            &dgb,
            &dbb,
            &dpb,
            &dal,
            &ddt,
            &part,
            rows as u32,
            heads as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();
    };
    bwd();
    let (mut wg, mut wbeta, mut wda, mut wdb) = (vec![], vec![], vec![], vec![]);
    let (mut wdal, mut wddt) = (vec![0.0f64; heads], vec![0.0f64; heads]);
    let (mut got_da, mut got_db) = (vec![], vec![]);
    let dp = dpb.read_f32();
    for r in 0..rows {
        for h in 0..heads {
            let (a, b) = (f64::from(p[r * ld + a_off + h]), f64::from(p[r * ld + b_off + h]));
            let (al, d) = (f64::from(a_log[h]), f64::from(dt[h]));
            let g = -al.exp() * softplus(a + d);
            let s = sigmoid(b);
            let o = r * heads + h;
            let da = f64::from(dg[o]) * -al.exp() * softplus_grad(a + d);
            wg.push(g);
            wbeta.push(s);
            wda.push(da);
            wdb.push(f64::from(dbeta[o]) * s * (1.0 - s));
            wdal[h] += f64::from(dg[o]) * g;
            wddt[h] += da;
            got_da.push(dp[r * ld + a_off + h]);
            got_db.push(dp[r * ld + b_off + h]);
        }
    }
    let label = format!("gates rows={rows} heads={heads}");
    close(&format!("{label} g"), &gb.read_f32(), &wg);
    close(&format!("{label} beta"), &betab.read_f32(), &wbeta);
    close(&format!("{label} da"), &got_da, &wda);
    close(&format!("{label} db"), &got_db, &wdb);
    assert_eq!(
        dp.iter().filter(|&&v| v != SENTINEL).count(),
        2 * rows * heads,
        "{label}: wrote outside the a and b columns"
    );
    let first = (dal.read_f32(), ddt.read_f32());
    close(&format!("{label} dA_log"), &first.0, &wdal);
    close(&format!("{label} ddt_bias"), &first.1, &wddt);
    bwd();
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        same(&dal.read_f32(), &first.0) && same(&ddt.read_f32(), &first.1),
        "{label}: changed on a rerun"
    );
}

#[test]
fn gdn_gates_forward_and_backward_match() {
    with_gpu(|rt| {
        for (rows, heads) in [(1, 1), (255, 3), (256, 4), (257, 2), (600, 16)] {
            run_gates(rt, rows, heads, rows as u64 * 3 + heads as u64);
        }
        // No rows: zero parameter gradients.
        let z = buf(rt, &[0.0; 64]);
        let logits = GdnGateLogits {
            buf: &z,
            ld: 8,
            a_off: 4,
            b_off: 0,
        };
        let params = GdnParams { a_log: &z, dt_bias: &z };
        let (dal, ddt) = (seeded(rt, 4, SENTINEL), seeded(rt, 4, SENTINEL));
        let dp = seeded(rt, 64, SENTINEL);
        gdn_gates_bwd(rt, &logits, &params, &z, &z, &dp, &dal, &ddt, &buf(rt, &[0.0; 1]), 0, 4).unwrap();
        rt.synchronize().unwrap();
        assert!(
            dal.read_f32().iter().chain(&ddt.read_f32()).all(|&v| v == 0.0),
            "rows = 0: gradients must be zeros"
        );
    });
}

#[test]
fn copy_cols_moves_exactly_its_window() {
    with_gpu(|rt| {
        let (rows, ld_s, ld_d, w) = (37usize, 50usize, 24usize, 20usize);
        let src = random_f32(rows * ld_s, 61);
        let sb = buf(rt, &src);
        let db = seeded(rt, rows * ld_d, SENTINEL);
        copy_cols(rt, win(&sb, ld_s, 17), win(&db, ld_d, 3), rows as u32, w as u32).unwrap();
        rt.synchronize().unwrap();
        let d = db.read_f32();
        for r in 0..rows {
            for c in 0..ld_d {
                let want = if (3..3 + w).contains(&c) {
                    src[r * ld_s + 17 + c - 3]
                } else {
                    SENTINEL
                };
                assert_eq!(d[r * ld_d + c].to_bits(), want.to_bits(), "row {r} col {c}");
            }
        }
        // Disjoint windows of one buffer are fine; overlapping ones are not.
        copy_cols(rt, win(&sb, ld_s, 0), win(&sb, ld_s, 25), rows as u32, w as u32).expect("disjoint windows");
        let m = copy_cols(rt, win(&sb, ld_s, 0), win(&sb, ld_s, 10), rows as u32, w as u32).unwrap_err();
        assert!(m.contains("dst overlaps src"), "{m}");
        let m = copy_cols(rt, win(&sb, ld_s, 0), win(&db, ld_d, 10), rows as u32, w as u32).unwrap_err();
        assert!(m.contains("copy_cols dst"), "{m}");
    });
}

#[test]
fn gdn_gates_refuse_bad_layouts() {
    with_gpu(|rt| {
        let z = buf(rt, &[0.0; 256]);
        let params = GdnParams { a_log: &z, dt_bias: &z };
        let (dal, ddt, dp, part) = (
            buf(rt, &[0.0; 4]),
            buf(rt, &[0.0; 4]),
            buf(rt, &[0.0; 256]),
            buf(rt, &[0.0; 8]),
        );
        let e = |r: Result<(), String>, needle: &str| {
            let m = r.expect_err(needle);
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        };
        let ok = GdnGateLogits {
            buf: &z,
            ld: 16,
            a_off: 4,
            b_off: 0,
        };
        gdn_gates_bwd(rt, &ok, &params, &z, &z, &dp, &dal, &ddt, &part, 8, 4).expect("a valid call");
        e(
            gdn_gates_bwd(
                rt,
                &GdnGateLogits { a_off: 2, ..ok },
                &params,
                &z,
                &z,
                &dp,
                &dal,
                &ddt,
                &part,
                8,
                4,
            ),
            "the a and b windows overlap",
        );
        e(
            gdn_gates_bwd(rt, &ok, &params, &z, &z, &dp, &dal, &ddt, &buf(rt, &[0.0; 7]), 8, 4),
            "part",
        );
        e(
            gdn_gates_bwd(rt, &ok, &params, &z, &z, &z, &dal, &ddt, &part, 8, 4),
            "dproj",
        );
        e(
            gdn_gates_bwd(
                rt,
                &GdnGateLogits { ld: 7, ..ok },
                &params,
                &z,
                &z,
                &dp,
                &dal,
                &ddt,
                &part,
                8,
                4,
            ),
            "a",
        );
        e(qwen35::gdn_gates(rt, &ok, &params, &z, &dal, 8, 4), "g");
    });
}

// ------------------------------------------------------------------ stress ---

/// Every backward kernel over randomly drawn shapes within its contract,
/// each checked as the targeted tests check it (f64 reference, writes only
/// in its window, bit-identical rerun). `TESSL_FUZZ_ITERS` / `TESSL_FUZZ_SEED`
/// scale it up (see `common::fuzz_plan`).
#[test]
fn randomized_shapes_stress() {
    let (iters, seed) = common::fuzz_plan(4);
    with_gpu(|rt| {
        let ws = EmbedBwdWorkspace::new(rt, 600).unwrap();
        for it in 0..iters {
            let s = seed.wrapping_mul(1_000_003).wrapping_add(it as u64);
            let mut r = common::SplitMix::new(s);
            eprintln!("stress iteration {it} (seed {s})");
            let (rows, d) = (r.range(1, 300), r.range(1, 4096));
            run_rms(rt, rows, d, r.range(0, 1) == 1, s);
            let (rows, heads, d) = (r.range(1, 150), r.range(1, 16), r.range(1, 512));
            run_gated(rt, rows, heads, d, s);
            let (b, t, c, kw) = (r.range(1, 3), r.range(1, 300), r.range(1, 700), r.range(2, 8));
            run_conv(rt, b, t, c, kw, s);
            let (hkv, group) = (r.range(1, 2), r.range(1, 4));
            let d = 2 * r.range(8, 256);
            let rot = 2 * r.range(0, d / 2);
            let theta = [1e4f32, 1e6, 1e7][r.range(0, 2)];
            run_qk(rt, r.range(1, 2), r.range(1, 80), hkv * group, hkv, d, rot, theta, s);
            run_gates(rt, r.range(1, 600), r.range(1, 16), s);
            let (n, vocab, hidden) = (r.range(1, 600), r.range(1, 300), r.range(1, 300));
            let ids: Vec<u32> = (0..n).map(|_| r.range(0, vocab - 1) as u32).collect();
            run_embed(rt, &ws, &ids, vocab, hidden, s);
        }
    });
}

/// Rows of `src` added into rows `pos` of `dst`, the other rows untouched,
/// one f32 add each; a repeated or out-of-range position is refused before
/// anything is written.
#[test]
fn scatter_add_rows_adds_each_row_at_its_position() {
    with_gpu(|rt| {
        let (rows, width) = (7usize, 37usize);
        let pos = [5u32, 0, 3];
        let src_v = random_f32(pos.len() * width, 11);
        let dst_v = random_f32(rows * width, 12);
        let src = rt.alloc_tensor_f32(&[pos.len(), width]).unwrap();
        src.buffer.write_f32(&src_v);
        let dst = rt.alloc_tensor_f32(&[rows, width]).unwrap();
        dst.buffer.write_f32(&dst_v);
        scatter_add_rows(rt, &src, &pos, &dst).unwrap();
        rt.synchronize().unwrap();
        let got = dst.read_f32().unwrap();
        let mut want = dst_v.clone();
        for (i, &p) in pos.iter().enumerate() {
            for c in 0..width {
                want[p as usize * width + c] += src_v[i * width + c];
            }
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&want));

        for (bad, needle) in [
            (&[1u32, 1, 2][..], "position 1 appears twice"),
            (&[0, 7, 2][..], "position 7 >= 7 rows"),
        ] {
            let m = scatter_add_rows(rt, &src, bad, &dst).unwrap_err();
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        }
        let m = scatter_add_rows(rt, &src, &pos[..2], &dst).unwrap_err();
        assert!(m.contains("src must be f32 [2, 37]"), "{m}");
        rt.synchronize().unwrap();
        assert_eq!(bits(&dst.read_f32().unwrap()), bits(&want), "a refusal wrote");
    });
}
