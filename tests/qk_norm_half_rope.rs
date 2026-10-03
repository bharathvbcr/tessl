//! Nanolab QK RMSNorm and half-split RoPE at head dim 64.
//!
//! `qwen35::attn_qk_norm_rope` still reads query head `j` at `j * 2 * D`
//! (Qwen packs `[q, gate]`) and multiplies by `(1 + w)`. The packed entry
//! takes the stride: `D` for `[T, H, D]`, `2 * D` for the old columns.

mod common;

use common::{buf, seeded, with_gpu};
use tessl::qwen35::{AttnShape, AttnTargets, Cols};

const D: usize = 64;
const H: usize = 2;
const T: usize = 3;
const EPS: f32 = 1e-6;
const THETA: f32 = 10_000.0;

fn shape(q_heads: u32, kv_heads: u32, rotary_dim: u32) -> AttnShape {
    AttnShape {
        batch: 1,
        seq: T as u32,
        q_heads,
        kv_heads,
        head_dim: D as u32,
        rotary_dim,
    }
}

fn row_sum_sq(x: &[f32]) -> f64 {
    x.iter()
        .map(|v| {
            let z = f64::from(*v);
            z * z
        })
        .sum()
}

/// `x * rsqrt(mean(x^2) + eps) * (bias + w)`, then half-split RoPE on the
/// whole head: `x * cos + cat(-x2, x1) * sin`.
fn norm_half_rope(x: &[f32], w: &[f32], bias: f32, pos: u32) -> Vec<f32> {
    let inv = 1.0 / (row_sum_sq(x) / D as f64 + f64::from(EPS)).sqrt();
    let mut n = vec![0.0f64; D];
    for i in 0..D {
        n[i] = f64::from(x[i]) * inv * (f64::from(bias) + f64::from(w[i]));
    }
    let half = D / 2;
    let mut y = vec![0.0f32; D];
    for p in 0..half {
        let inv_freq = 1.0 / f64::from(THETA).powf((2 * p) as f64 / D as f64);
        let angle = f64::from(pos) * inv_freq;
        let (s, c) = angle.sin_cos();
        let x1 = n[p];
        let x2 = n[p + half];
        y[p] = (x1 * c - x2 * s) as f32;
        y[p + half] = (x2 * c + x1 * s) as f32;
    }
    y
}

fn pattern(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.17 + seed).sin()) * 0.4 + 0.05).collect()
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what} [{i}] non-finite {g}");
        let tol = 2e-4 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{what} [{i}] got {g} want {w} tol {tol}");
    }
}

#[test]
fn packed_stride_applies_qk_norm_and_half_split_rope() {
    with_gpu(|rt| {
        let heads = H as u32;
        let src = pattern(T * H * D, 0.3);
        let qw = pattern(D, 1.1);
        let kw = pattern(D, 2.4);
        let src_b = buf(rt, &src);
        let q = seeded(rt, T * H * D, 4.5e28);
        let k = seeded(rt, T * H * D, 4.5e28);
        let v = seeded(rt, T * H * D, 4.5e28);
        tessl::qwen35::attn_qk_norm_rope_packed(
            rt,
            &shape(heads, heads, D as u32),
            Cols::dense(&src_b, (H * D) as u32),
            D as u32,
            0.0,
            &buf(rt, &qw),
            &buf(rt, &kw),
            &AttnTargets {
                q_out: &q,
                k_cache: &k,
                v_cache: &v,
            },
            0,
            THETA,
            EPS,
        )
        .expect("packed QK-norm");
        rt.synchronize().expect("sync");

        let mut want_q = vec![0.0f32; T * H * D];
        let mut want_k = vec![0.0f32; T * H * D];
        for t in 0..T {
            for h in 0..H {
                let at = (t * H + h) * D;
                let row = &src[at..at + D];
                want_q[at..at + D].copy_from_slice(&norm_half_rope(row, &qw, 0.0, t as u32));
                want_k[at..at + D].copy_from_slice(&norm_half_rope(row, &kw, 0.0, t as u32));
            }
        }
        close(&q.read_f32()[..T * H * D], &want_q, "q");
        close(&k.read_f32()[..T * H * D], &want_k, "k");
        close(&v.read_f32()[..T * H * D], &src, "v");
    });
}

#[test]
fn query_stride_2d_skips_interleaved_gate_columns() {
    with_gpu(|rt| {
        // Two query heads, Qwen spacing: head j occupies `j * 2D` and the
        // next D columns are a gate the kernel must not read. One KV head
        // sits at column 0, the same place as query head 0.
        let width = 4 * D;
        let mut src = vec![0.2f32; T * width];
        let mut want_q = vec![0.0f32; T * H * D];
        let qw = pattern(D, 0.7);
        let kw = pattern(D, 1.3);
        for t in 0..T {
            let row = t * width;
            let head0 = pattern(D, 3.0 + t as f32);
            let gate = vec![50.0f32; D];
            let head1 = pattern(D, 8.0 + t as f32);
            src[row..row + D].copy_from_slice(&head0);
            src[row + D..row + 2 * D].copy_from_slice(&gate);
            src[row + 2 * D..row + 3 * D].copy_from_slice(&head1);
            let q0 = norm_half_rope(&head0, &qw, 1.0, t as u32);
            let q1 = norm_half_rope(&head1, &qw, 1.0, t as u32);
            want_q[(t * H) * D..(t * H) * D + D].copy_from_slice(&q0);
            want_q[(t * H + 1) * D..(t * H + 2) * D].copy_from_slice(&q1);
        }
        let src_b = buf(rt, &src);
        let q = seeded(rt, T * H * D, 7.5e28);
        let k = seeded(rt, T * D, 7.5e28);
        let v = seeded(rt, T * D, 7.5e28);
        tessl::qwen35::attn_qk_norm_rope_packed(
            rt,
            &shape(H as u32, 1, D as u32),
            Cols::dense(&src_b, width as u32),
            (2 * D) as u32,
            1.0,
            &buf(rt, &qw),
            &buf(rt, &kw),
            &AttnTargets {
                q_out: &q,
                k_cache: &k,
                v_cache: &v,
            },
            0,
            THETA,
            EPS,
        )
        .expect("stride 2*D");
        rt.synchronize().expect("sync");
        close(&q.read_f32()[..T * H * D], &want_q, "q stride 2D");

        let err = tessl::qwen35::attn_qk_norm_rope_packed(
            rt,
            &shape(H as u32, 1, D as u32),
            Cols::dense(&src_b, width as u32),
            (D - 1) as u32,
            0.0,
            &buf(rt, &qw),
            &buf(rt, &kw),
            &AttnTargets {
                q_out: &q,
                k_cache: &k,
                v_cache: &v,
            },
            0,
            THETA,
            EPS,
        )
        .expect_err("stride shorter than a head");
        assert!(err.contains("shorter than head_dim"), "{err}");
    });
}
