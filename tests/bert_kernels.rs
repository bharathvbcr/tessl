//! The BERT encoder's kernels (`tessl::bert`, `kernels/bert.metal`) and the
//! head dims 32 and 64 of `embedgemma2::encoder_attn`, each against an f64
//! reference.
//!
//! Bounds, fixed before any run:
//!
//! * LayerNorm (plain, fused, and after the embedding sum) within
//!   `1e-5 * max(1, max|ref|)`: one f32 rounding per term of two 384-3072
//!   term sums, then one multiply-add;
//! * the erf GELU within `(1e-6 + 4 x^2 eps) * |ref| + 1e-30` — *relative*,
//!   element by element, so the left tail is judged on its own digits. The
//!   `x^2` term is f32's own conditioning there, not slack: rounding `x/sqrt 2`
//!   and `z^2` puts a relative error of about `x^2 eps` into the exponent of
//!   `exp(-z^2)`, which no f32 erfc can avoid (torch's f32 GELU gives exactly
//!   0 past x = -5.8, where `1 + erf` rounds to 0). The tanh approximation
//!   (`nn::mlp_gelu_tanh`) must fail the same bound, or the test is not able
//!   to tell the two apart;
//! * the sparse max within `1e-6 * max(1, |ref|)` per element: a max is exact,
//!   so only the bias add and one log1p round;
//! * attention within `2e-5 * max|ref|`, `embedgemma2_kernels.rs`'s bound.
//!
//! Adversarial: a zero-variance row (must be exactly the bias), a row with a
//! mean of 1e4 (a one-pass variance cancels there), widths of 1 and past 1024
//! lanes, all-negative logits (pooled exactly 0), padding rows holding huge
//! logits outside every segment (never pooled), the 30522-wide vocabulary
//! (not a multiple of any launch width), an id past the vocabulary (its row
//! NaN, the rest intact), and every host-side refusal.

mod common;

use std::sync::Arc;

use common::{with_gpu, SplitMix};
use tessl::bert::{
    bias_add, bias_gelu_erf, bias_residual_layer_norm, embed_layer_norm, layer_norm, segment_sparse_max,
    upload_segments, BertConfig, BertFamily, EmbedTables,
};
use tessl::embedgemma2::{encoder_attn, EncoderAttnDims};
use tessl::nn::mlp_gelu_tanh;
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

fn buf_f32(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).unwrap();
    b.write_f32(data);
    b
}

fn buf_u32(rt: &Arc<GpuRuntime>, data: &[u32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).unwrap();
    b.write_u32(data);
    b
}

fn read(rt: &Arc<GpuRuntime>, b: &GpuBuffer, n: usize) -> Vec<f32> {
    rt.synchronize().unwrap();
    b.read_f32()[..n].to_vec()
}

fn rand_vec(rng: &mut SplitMix, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.unit() * scale).collect()
}

/// `max |got - want|` against `rel * max(1, max |want|)`.
fn assert_abs(what: &str, got: &[f32], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0f64, |m, &w| m.max(w.abs()));
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let e = (f64::from(g) - w).abs();
        if !e.is_finite() || e > worst.0 {
            worst = (if e.is_finite() { e } else { f64::INFINITY }, i);
        }
    }
    assert!(
        worst.0 <= rel * scale,
        "{what}: max error {:.3e} at {} exceeds {rel:.0e} * {scale:.3e} (got {}, want {})",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
}

// ---------------------------------------------------------------------------
// f64 references
// ---------------------------------------------------------------------------

fn layer_norm_ref(x: &[f64], w: &[f32], b: &[f32], eps: f64) -> Vec<f64> {
    let d = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(d) {
        let mean = row.iter().sum::<f64>() / d as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + eps).sqrt();
        out.extend(
            row.iter()
                .enumerate()
                .map(|(i, v)| (v - mean) * inv * f64::from(w[i]) + f64::from(b[i])),
        );
    }
    out
}

/// erfc in f64: the Maclaurin series of erf below 2.5 (about 1e-14 after its
/// alternating terms cancel), the continued fraction above (Lentz), which
/// keeps relative accuracy far into the tail.
fn erfc_f64(x: f64) -> f64 {
    let z = x.abs();
    let r = if z < 2.5 {
        let (mut term, mut sum, mut n) = (z, z, 0.0f64);
        loop {
            n += 1.0;
            term *= -z * z / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() || n > 200.0 {
                break;
            }
        }
        1.0 - sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc(z) = exp(-z^2)/sqrt(pi) * 1/(z + 1/2/(z + 1/(z + 3/2/(z + ...)))).
        let tiny = 1e-300;
        let mut f = z;
        let (mut c, mut d) = (z, 0.0f64);
        for k in 1..400 {
            let a = k as f64 / 2.0;
            d = z + a * d;
            d = if d.abs() < tiny { tiny } else { d };
            c = z + a / c;
            c = if c.abs() < tiny { tiny } else { c };
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (-z * z).exp() / std::f64::consts::PI.sqrt() / f
    };
    if x >= 0.0 {
        r
    } else {
        2.0 - r
    }
}

fn gelu_erf_ref(x: f64) -> f64 {
    0.5 * x * erfc_f64(-x / std::f64::consts::SQRT_2)
}

/// f32's unit roundoff.
const EPS32: f64 = 5.960_464_477_539_063e-8;

/// Element-wise relative error against `(rel + 4 x^2 eps) * |want| + 1e-30`
/// for inputs `x`; returns the worst offender so a test can require failure
/// as well as success.
fn worst_relative(x: &[f32], got: &[f32], want: &[f64], rel: f64) -> Option<(usize, f64)> {
    let mut worst: Option<(usize, f64)> = None;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let e = (f64::from(g) - w).abs();
        let xi = f64::from(x[i]);
        let allowed = (rel + 4.0 * xi * xi * EPS32) * w.abs() + 1e-30;
        // A NaN error or bound is a failure too.
        if !matches!(
            e.partial_cmp(&allowed),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            let ratio = if e.is_finite() { e / allowed } else { f64::INFINITY };
            if worst.is_none_or(|(_, r)| ratio > r) {
                worst = Some((i, ratio));
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Reference self-checks (CPU only)
// ---------------------------------------------------------------------------

#[test]
fn the_erfc_reference_matches_known_values() {
    // Abramowitz and Stegun table 7.1 and their asymptotic tail.
    for (x, want) in [
        (0.0, 1.0),
        (0.5, 0.479_500_122_186_953_5),
        (1.0, 0.157_299_207_050_285_1),
        (2.0, 4.677_734_981_047_266e-3),
        (2.5, 4.069_520_174_449_59e-4),
        (3.0, 2.209_049_699_858_544e-5),
        (5.0, 1.537_459_794_428_035e-12),
        (10.0, 2.088_487_583_762_545e-45),
        (-1.0, 1.842_700_792_949_715),
    ] {
        let got = erfc_f64(x);
        assert!(
            ((got - want) / want).abs() < 1e-12,
            "erfc({x}) = {got:e}, want {want:e}"
        );
    }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

#[test]
fn layer_norm_matches_f64_including_adversarial_rows() {
    with_gpu(|rt| {
        let mut rng = SplitMix::new(11);
        for &dim in &[1usize, 7, 32, 384, 768, 1025, 3072] {
            let rows = 9usize;
            let mut x = rand_vec(&mut rng, rows * dim, 3.0);
            // Row 0: zero variance. Row 1: a huge mean and a small spread,
            // which a one-pass E[x^2] - E[x]^2 loses entirely.
            for v in &mut x[..dim] {
                *v = 2.5;
            }
            for (i, v) in x[dim..2 * dim].iter_mut().enumerate() {
                *v = 1.0e4 + ((i % 5) as f32 - 2.0);
            }
            let w = rand_vec(&mut rng, dim, 2.0);
            let b = rand_vec(&mut rng, dim, 1.0);
            let (xb, wb, bb) = (buf_f32(rt, &x), buf_f32(rt, &w), buf_f32(rt, &b));
            let out = rt.alloc_buffer(rows * dim * 4).unwrap();
            layer_norm(rt, &xb, &wb, &bb, &out, rows as u32, dim as u32, 1e-12).unwrap();
            let got = read(rt, &out, rows * dim);
            let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
            let want = layer_norm_ref(&x64, &w, &b, 1e-12);
            // The huge-mean row is judged apart: its deviations (+-2) sit
            // under a mean f32 resolves to ~1e-3, so 5e-3 is two-pass f32's
            // floor there, while a one-pass E[x^2] - E[x]^2 (ulp 8 at 1e8)
            // returns noise or NaN.
            assert_abs(
                &format!("layer_norm dim {dim} rows 2.."),
                &got[2 * dim..],
                &want[2 * dim..],
                1e-5,
            );
            assert_eq!(
                &got[..dim],
                &b[..],
                "dim {dim}: a zero-variance row is exactly the bias"
            );
            if dim > 1 {
                let x32: Vec<f64> = x[dim..2 * dim].iter().map(|&v| f64::from(v)).collect();
                let want1 = layer_norm_ref(&x32, &w, &b, 1e-12);
                assert_abs(
                    &format!("layer_norm dim {dim} huge-mean row"),
                    &got[dim..2 * dim],
                    &want1,
                    5e-3,
                );
            }
        }
    });
}

#[test]
fn the_fused_bias_residual_layer_norm_matches_f64_in_place() {
    with_gpu(|rt| {
        let mut rng = SplitMix::new(12);
        for &(rows, dim) in &[(1usize, 384usize), (37, 384), (5, 768), (3, 1025)] {
            let y = rand_vec(&mut rng, rows * dim, 4.0);
            let resid = rand_vec(&mut rng, rows * dim, 4.0);
            let bias = rand_vec(&mut rng, dim, 1.0);
            let w = rand_vec(&mut rng, dim, 2.0);
            let b = rand_vec(&mut rng, dim, 1.0);
            let rb = buf_f32(rt, &resid);
            let (yb, biasb, wb, bb) = (buf_f32(rt, &y), buf_f32(rt, &bias), buf_f32(rt, &w), buf_f32(rt, &b));
            bias_residual_layer_norm(rt, &yb, &biasb, &rb, &wb, &bb, rows as u32, dim as u32, 1e-12).unwrap();
            let got = read(rt, &rb, rows * dim);
            let sum: Vec<f64> = (0..rows * dim)
                .map(|i| f64::from(y[i]) + f64::from(bias[i % dim]) + f64::from(resid[i]))
                .collect();
            assert_abs(
                &format!("fused LN {rows}x{dim}"),
                &got,
                &layer_norm_ref(&sum, &w, &b, 1e-12),
                1e-5,
            );
        }
    });
}

#[test]
fn the_embedding_sum_matches_f64_and_a_bad_id_poisons_only_its_row() {
    with_gpu(|rt| {
        let mut rng = SplitMix::new(13);
        let (vocab, positions, dim, seq, batch) = (1000usize, 64usize, 384usize, 40usize, 3usize);
        let word = rand_vec(&mut rng, vocab * dim, 1.0);
        let pos = rand_vec(&mut rng, positions * dim, 0.5);
        let ty = rand_vec(&mut rng, dim, 0.25);
        let w = rand_vec(&mut rng, dim, 2.0);
        let b = rand_vec(&mut rng, dim, 1.0);
        let rows = batch * seq;
        let mut ids: Vec<u32> = (0..rows).map(|_| rng.range(0, vocab - 1) as u32).collect();
        ids[45] = vocab as u32 + 7;
        let bufs = (
            buf_f32(rt, &word),
            buf_f32(rt, &pos),
            buf_f32(rt, &ty),
            buf_f32(rt, &w),
            buf_f32(rt, &b),
        );
        let tables = EmbedTables {
            word: &bufs.0,
            vocab: vocab as u32,
            pos: &bufs.1,
            positions: positions as u32,
            type_row: &bufs.2,
            ln_w: &bufs.3,
            ln_b: &bufs.4,
        };
        let idb = buf_u32(rt, &ids);
        let out = rt.alloc_buffer(rows * dim * 4).unwrap();
        embed_layer_norm(rt, &idb, tables, &out, rows as u32, dim as u32, seq as u32, 1e-12).unwrap();
        let got = read(rt, &out, rows * dim);
        assert!(
            got[45 * dim..46 * dim].iter().all(|v| v.is_nan()),
            "an id past the vocabulary is NaN"
        );
        let mut sum = Vec::with_capacity(rows * dim);
        for (r, &id) in ids.iter().enumerate() {
            let id = (id as usize).min(vocab - 1);
            let t = r % seq;
            sum.extend(
                (0..dim).map(|d| f64::from(word[id * dim + d]) + f64::from(pos[t * dim + d]) + f64::from(ty[d])),
            );
        }
        let want = layer_norm_ref(&sum, &w, &b, 1e-12);
        let keep: Vec<usize> = (0..rows).filter(|&r| r != 45).collect();
        let g: Vec<f32> = keep
            .iter()
            .flat_map(|&r| got[r * dim..(r + 1) * dim].to_vec())
            .collect();
        let wv: Vec<f64> = keep
            .iter()
            .flat_map(|&r| want[r * dim..(r + 1) * dim].to_vec())
            .collect();
        assert_abs("embedding sum", &g, &wv, 1e-5);

        // Refusals: a sequence past the position table, rows not a whole
        // number of sequences, a bad eps, a short buffer.
        let e = embed_layer_norm(
            rt,
            &idb,
            tables,
            &out,
            rows as u32,
            dim as u32,
            positions as u32 + 1,
            1e-12,
        );
        assert!(e.unwrap_err().contains("position table"));
        let e = embed_layer_norm(rt, &idb, tables, &out, rows as u32 - 1, dim as u32, seq as u32, 1e-12);
        assert!(e.unwrap_err().contains("whole number"));
        let e = embed_layer_norm(rt, &idb, tables, &out, rows as u32, dim as u32, seq as u32, 0.0);
        assert!(e.unwrap_err().contains("eps"));
        let small = rt.alloc_buffer(4).unwrap();
        assert!(embed_layer_norm(rt, &idb, tables, &small, rows as u32, dim as u32, seq as u32, 1e-12).is_err());
    });
}

// ---------------------------------------------------------------------------
// Bias add and GELU
// ---------------------------------------------------------------------------

#[test]
fn bias_add_is_exact() {
    with_gpu(|rt| {
        let mut rng = SplitMix::new(14);
        let (rows, cols) = (17usize, 30522usize);
        let x = rand_vec(&mut rng, rows * cols, 8.0);
        let bias = rand_vec(&mut rng, cols, 2.0);
        let xb = buf_f32(rt, &x);
        bias_add(rt, &xb, &buf_f32(rt, &bias), rows as u32, cols as u32).unwrap();
        let got = read(rt, &xb, rows * cols);
        for (i, &g) in got.iter().enumerate() {
            assert_eq!(g, x[i] + bias[i % cols], "element {i}");
        }
        assert!(
            bias_add(rt, &xb, &buf_f32(rt, &bias), rows as u32, 0).is_err(),
            "cols 0 is refused"
        );
        assert!(
            bias_add(rt, &xb, &xb, rows as u32, cols as u32).is_err(),
            "x aliasing bias is refused"
        );
    });
}

#[test]
fn the_erf_gelu_matches_f64_and_the_tanh_form_does_not() {
    with_gpu(|rt| {
        // A dense sweep of the range where the two forms differ, both tails,
        // zero, and values past where either saturates.
        let mut x: Vec<f32> = (-4000..=4000).map(|i| i as f32 * 0.003).collect();
        x.extend([
            -30.0, -15.0, -12.0, -9.5, -8.0, 0.0, 1e-30, -1e-30, 8.0, 12.0, 50.0, 1e6, -1e6,
        ]);
        let n = x.len();
        let bias = vec![0.0f32; 1];
        let xb = buf_f32(rt, &x);
        bias_gelu_erf(rt, &xb, &buf_f32(rt, &bias), n as u32, 1).unwrap();
        let got = read(rt, &xb, n);
        let want: Vec<f64> = x.iter().map(|&v| gelu_erf_ref(f64::from(v))).collect();
        if let Some((i, ratio)) = worst_relative(&x, &got, &want, 1e-6) {
            panic!(
                "gelu_erf({}) = {}, want {:e} ({ratio:.1}x the bound)",
                x[i], got[i], want[i]
            );
        }

        // The same bound must reject the tanh approximation, or this test
        // cannot tell the kernel computes erf at all.
        let ones = buf_f32(rt, &vec![1.0f32; n]);
        let tanh_out = rt.alloc_buffer(n * 4).unwrap();
        mlp_gelu_tanh(rt, &buf_f32(rt, &x), &ones, &tanh_out, n as u32).unwrap();
        let tanh = read(rt, &tanh_out, n);
        assert!(
            worst_relative(&x, &tanh, &want, 1e-6).is_some(),
            "the tanh GELU passed the erf bound, so the bound does not distinguish them"
        );

        // With a bias, per column.
        let (rows, cols) = (3usize, 1536usize);
        let mut rng = SplitMix::new(15);
        let y = rand_vec(&mut rng, rows * cols, 6.0);
        let bias = rand_vec(&mut rng, cols, 1.0);
        let yb = buf_f32(rt, &y);
        bias_gelu_erf(rt, &yb, &buf_f32(rt, &bias), rows as u32, cols as u32).unwrap();
        let got = read(rt, &yb, rows * cols);
        let biased: Vec<f32> = (0..rows * cols).map(|i| y[i] + bias[i % cols]).collect();
        let want: Vec<f64> = biased.iter().map(|&v| gelu_erf_ref(f64::from(v))).collect();
        if let Some((i, ratio)) = worst_relative(&biased, &got, &want, 1e-6) {
            panic!("bias_gelu_erf element {i}: {} vs {:e} ({ratio:.1}x)", got[i], want[i]);
        }
    });
}

// ---------------------------------------------------------------------------
// Segment sparse max
// ---------------------------------------------------------------------------

fn sparse_ref(logits: &[f32], bias: &[f32], segs: &[(u32, u32)], v: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; segs.len() * v];
    for (s, &(a, b)) in segs.iter().enumerate() {
        for c in 0..v {
            let mut m = 0.0f64;
            for r in a..b {
                let z = f64::from(logits[r as usize * v + c] + bias[c]);
                m = m.max(z.max(0.0).ln_1p());
            }
            out[s * v + c] = m;
        }
    }
    out
}

#[test]
fn the_segment_sparse_max_matches_f64_and_ignores_rows_outside_segments() {
    with_gpu(|rt| {
        let mut rng = SplitMix::new(16);
        let v = 30522usize;
        let rows = 20usize;
        let mut logits = rand_vec(&mut rng, rows * v, 12.0);
        // Rows 5..8 and 17..20 are padding: huge logits no segment covers.
        for r in (5..8).chain(17..20) {
            for c in 0..v {
                logits[r * v + c] = 1.0e6;
            }
        }
        // Segment 2 is all-negative logits: its vector must be exactly zero.
        for r in 12..17 {
            for c in 0..v {
                logits[r * v + c] = -1.0 - (c % 7) as f32;
            }
        }
        let bias = rand_vec(&mut rng, v, 0.5);
        let segs = vec![(0u32, 5u32), (8, 12), (12, 17)];
        let lb = buf_f32(rt, &logits);
        let bb = buf_f32(rt, &bias);
        let sb = upload_segments(rt, &segs, rows as u32).unwrap();
        let pooled = rt.alloc_buffer(segs.len() * v * 4).unwrap();
        pooled.zero();
        segment_sparse_max(rt, &lb, &bb, &sb, &pooled, segs.len() as u32, rows as u32, v as u32).unwrap();
        let got = read(rt, &pooled, segs.len() * v);
        let want = sparse_ref(&logits, &bias, &segs, v);
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!((f64::from(g) - w).abs() <= 1e-6 * w.max(1.0), "element {i}: {g} vs {w}");
        }
        // Segment 2's logits are <= -1 and the bias is within +-0.5, so every
        // biased logit is negative and the whole vector is exactly 0.
        assert!(
            got[2 * v..].iter().all(|&g| g == 0.0),
            "all-negative logits pool to exactly 0"
        );
        assert!(
            got.iter().all(|&g| g < 20.0),
            "a padding row's 1e6 logit leaked into a segment"
        );

        // It accumulates: pooling segment 0 in two halves equals pooling it once.
        let halves = upload_segments(rt, &[(0, 2)], rows as u32).unwrap();
        let rest = upload_segments(rt, &[(2, 5)], rows as u32).unwrap();
        let acc = rt.alloc_buffer(v * 4).unwrap();
        acc.zero();
        segment_sparse_max(rt, &lb, &bb, &halves, &acc, 1, rows as u32, v as u32).unwrap();
        segment_sparse_max(rt, &lb, &bb, &rest, &acc, 1, rows as u32, v as u32).unwrap();
        assert_eq!(read(rt, &acc, v), got[..v].to_vec(), "two passes equal one");

        // Refusals.
        assert!(
            upload_segments(rt, &[(3, 3)], rows as u32).is_err(),
            "an empty segment is refused"
        );
        assert!(
            upload_segments(rt, &[(0, 21)], rows as u32).is_err(),
            "a segment past the rows is refused"
        );
        assert!(
            segment_sparse_max(rt, &lb, &bb, &sb, &lb, 3, rows as u32, v as u32).is_err(),
            "pooled aliasing logits"
        );
        assert!(
            segment_sparse_max(rt, &lb, &bb, &sb, &pooled, 3, rows as u32, 0).is_err(),
            "vocab 0"
        );
    });
}

// ---------------------------------------------------------------------------
// Attention at BERT's head dims
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn attn_ref(b: usize, t: usize, h: usize, d: usize, lens: &[usize], q: &[f32], k: &[f32], v: &[f32]) -> Vec<f64> {
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0f64; b * t * h * d];
    let at = |bb: usize, tt: usize, hh: usize| ((bb * t + tt) * h + hh) * d;
    for bb in 0..b {
        for hh in 0..h {
            for i in 0..lens[bb] {
                let s: Vec<f64> = (0..lens[bb])
                    .map(|j| {
                        (0..d)
                            .map(|x| f64::from(q[at(bb, i, hh) + x]) * f64::from(k[at(bb, j, hh) + x]))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for x in 0..d {
                    out[at(bb, i, hh) + x] = (0..lens[bb])
                        .map(|j| e[j] * f64::from(v[at(bb, j, hh) + x]))
                        .sum::<f64>()
                        / z;
                }
            }
        }
    }
    out
}

#[test]
fn encoder_attention_at_head_dims_32_and_64_matches_f64() {
    with_gpu(|rt| {
        let mut rng = SplitMix::new(17);
        for &(d, h) in &[(32usize, 12usize), (64, 12)] {
            for &(b, t, ref lens) in &[
                (1usize, 1usize, vec![1usize]),
                (3, 130, vec![130, 1, 77]),
                (2, 512, vec![512, 300]),
            ] {
                let n = b * t * h * d;
                let q = rand_vec(&mut rng, n, 2.0);
                let mut k = rand_vec(&mut rng, n, 2.0);
                let mut v = rand_vec(&mut rng, n, 1.0);
                // Padding positions hold huge keys and values: one unmasked
                // padded key would swamp every live row's softmax.
                for (bb, &len) in lens.iter().enumerate() {
                    for i in ((bb * t + len) * h * d)..((bb + 1) * t * h * d) {
                        k[i] = 1.0e6;
                        v[i] = 1.0e6;
                    }
                }
                let (qb, kb, vb) = (buf_f32(rt, &q), buf_f32(rt, &k), buf_f32(rt, &v));
                let ob = rt.alloc_buffer(n * 4).unwrap();
                let lb = buf_u32(rt, &lens.iter().map(|&l| l as u32).collect::<Vec<_>>());
                let dims = EncoderAttnDims {
                    batch: b as u32,
                    seq: t as u32,
                    heads: h as u32,
                    heads_kv: h as u32,
                    head_dim: d as u32,
                    window: 0,
                    scale: 1.0 / (d as f32).sqrt(),
                };
                encoder_attn(rt, &qb, &kb, &vb, &ob, &lb, dims, false).unwrap();
                let got = read(rt, &ob, n);
                let want = attn_ref(b, t, h, d, lens, &q, &k, &v);
                let scale = want.iter().fold(0.0f64, |m, &w| m.max(w.abs()));
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (f64::from(g) - w).abs() <= 2e-5 * scale,
                        "d {d} b {b} t {t}: element {i}: {g} vs {w}"
                    );
                }
                // Padded queries are zeros (the reference leaves them zero).
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const MINI: &str = r#"{"architectures":["BertForMaskedLM"],"hidden_act":"gelu","hidden_size":384,
"intermediate_size":1536,"layer_norm_eps":1e-12,"max_position_embeddings":512,"model_type":"bert",
"num_attention_heads":12,"num_hidden_layers":6,"position_embedding_type":"absolute","type_vocab_size":2,
"vocab_size":30522}"#;

const DISTIL: &str = r#"{"activation":"gelu","architectures":["DistilBertForMaskedLM"],"dim":768,
"hidden_dim":3072,"max_position_embeddings":512,"model_type":"distilbert","n_heads":12,"n_layers":6,
"sinusoidal_pos_embds":false,"vocab_size":30522}"#;

#[test]
fn the_config_parser_reads_both_families_and_refuses_what_it_cannot_run() {
    let c = BertConfig::from_config_json(MINI).unwrap();
    assert_eq!(
        (c.family, c.hidden, c.layers, c.heads, c.head_dim()),
        (BertFamily::Bert, 384, 6, 12, 32)
    );
    assert_eq!(
        (c.intermediate, c.vocab, c.max_positions, c.type_vocab),
        (1536, 30522, 512, 2)
    );
    assert_eq!(c.layer_norm_eps, 1e-12);
    let c = BertConfig::from_config_json(DISTIL).unwrap();
    assert_eq!(
        (c.family, c.hidden, c.head_dim(), c.intermediate, c.type_vocab),
        (BertFamily::DistilBert, 768, 64, 3072, 0)
    );
    assert_eq!(c.layer_norm_eps, 1e-12);

    for (edit, needle) in [
        ((r#""hidden_act":"gelu""#, r#""hidden_act":"gelu_new""#), "gelu"),
        (
            (
                r#""position_embedding_type":"absolute""#,
                r#""position_embedding_type":"relative_key""#,
            ),
            "absolute",
        ),
        ((r#""model_type":"bert""#, r#""model_type":"roberta""#), "model_type"),
        (
            (r#""num_attention_heads":12"#, r#""num_attention_heads":5"#),
            "multiple",
        ),
        ((r#""hidden_size":384"#, r#""hidden_size":0"#), "zero"),
        ((r#""layer_norm_eps":1e-12"#, r#""layer_norm_eps":0"#), "layer_norm_eps"),
        ((r#""vocab_size":30522"#, r#""vocab_size":-1"#), "vocab_size"),
    ] {
        let text = MINI.replace(edit.0, edit.1);
        assert_ne!(text, MINI, "the edit {edit:?} must apply");
        let err = BertConfig::from_config_json(&text).map(|_| ()).unwrap_err();
        assert!(err.contains(needle), "{edit:?}: {err}");
    }
    // 6 heads over 384 is head dim 64, which has a kernel; 24 is 16, which
    // does not.
    let six = MINI.replace(r#""num_attention_heads":12"#, r#""num_attention_heads":6"#);
    assert_eq!(BertConfig::from_config_json(&six).unwrap().head_dim(), 64);
    let many = MINI.replace(r#""num_attention_heads":12"#, r#""num_attention_heads":24"#);
    assert!(BertConfig::from_config_json(&many).unwrap_err().contains("head dim 16"));
    let sin = DISTIL.replace(r#""sinusoidal_pos_embds":false"#, r#""sinusoidal_pos_embds":true"#);
    assert!(BertConfig::from_config_json(&sin).unwrap_err().contains("sinusoidal"));
    assert!(BertConfig::from_config_json("[]").is_err());
}
