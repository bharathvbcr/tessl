//! f64 references for EmbeddingGemma 2. `tests/embedgemma2_kernels.rs` holds
//! each one to transformers by fixture (`scripts/gen_embedgemma2_fixtures.py`);
//! `tests/embedgemma2_tiny.rs` composes them into a whole host forward.

/// `rms_norm_eps` of the checkpoint, used by every reference here.
pub const EPS: f64 = 1e-6;

pub struct Attn {
    pub b: usize,
    pub t: usize,
    pub h: usize,
    pub hkv: usize,
    pub d: usize,
    pub window: usize,
    pub lens: Vec<usize>,
}

/// Bidirectional attention, scale 1.0, transformers' masking: key j is visible
/// to query i of row b iff j < lens[b] and (window == 0 or |i - j| <= window).
/// Queries at or past lens[b] are reported as zeros (the kernel's contract;
/// transformers computes them over padding and nothing reads them).
pub fn attn_ref(a: &Attn, q: &[f64], k: &[f64], v: &[f64], rows: Option<&[usize]>) -> Vec<f64> {
    let mut out = vec![0.0; a.b * a.t * a.h * a.d];
    let group = a.h / a.hkv;
    let all: Vec<usize> = (0..a.t).collect();
    let rows = rows.unwrap_or(&all);
    for b in 0..a.b {
        let len = a.lens[b].min(a.t);
        for &i in rows {
            if i >= len {
                continue;
            }
            let lo = if a.window == 0 { 0 } else { i.saturating_sub(a.window) };
            let hi = if a.window == 0 { len } else { len.min(i + a.window + 1) };
            for h in 0..a.h {
                let hk = h / group;
                let qo = ((b * a.t + i) * a.h + h) * a.d;
                let scores: Vec<f64> = (lo..hi)
                    .map(|j| {
                        let ko = ((b * a.t + j) * a.hkv + hk) * a.d;
                        (0..a.d).map(|x| q[qo + x] * k[ko + x]).sum::<f64>()
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = w.iter().sum();
                for (n, j) in (lo..hi).enumerate() {
                    let vo = ((b * a.t + j) * a.hkv + hk) * a.d;
                    for x in 0..a.d {
                        out[qo + x] += w[n] / z * v[vo + x];
                    }
                }
            }
        }
    }
    out
}

/// `x / sqrt(mean(x^2) + EPS) * w` (`* w`, not `* (1 + w)`), or no weight.
pub fn rms(x: &[f64], w: Option<&[f64]>) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + EPS).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| v * inv * w.map_or(1.0, |w| w[i]))
        .collect()
}

/// Full-width rotate_half RoPE at `pos`, `inv_freq = theta^(-2p/d)`.
pub fn rope(x: &[f64], pos: f64, theta: f64) -> Vec<f64> {
    let d = x.len();
    let half = d / 2;
    let mut out = vec![0.0; d];
    for p in 0..half {
        let f = pos / theta.powf(2.0 * p as f64 / d as f64);
        let (s, c) = f.sin_cos();
        out[p] = x[p] * c - x[p + half] * s;
        out[p + half] = x[p + half] * c + x[p] * s;
    }
    out
}

pub fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// `nn.Linear` without bias: `w` is `[out_f, in_f]`, row-major.
pub fn lin(w: &[f64], x: &[f64], out_f: usize, in_f: usize) -> Vec<f64> {
    (0..out_f)
        .map(|o| (0..in_f).map(|i| w[o * in_f + i] * x[i]).sum())
        .collect()
}

/// Mean over each sequence's live rows of `x: [b, t, d]`, then L2 normalize.
pub fn pool_reference(x: &[f64], lens: &[usize], t: usize, d: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for (b, &len) in lens.iter().enumerate() {
        let mut mean = vec![0.0; d];
        for ti in 0..len {
            for (j, m) in mean.iter_mut().enumerate() {
                *m += x[(b * t + ti) * d + j];
            }
        }
        let mean: Vec<f64> = mean.iter().map(|v| v / len as f64).collect();
        out.extend(l2_normalize(&mean));
    }
    out
}

/// `x / max(||x||, 1e-12)`.
pub fn l2_normalize(x: &[f64]) -> Vec<f64> {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    x.iter().map(|v| v / norm).collect()
}
