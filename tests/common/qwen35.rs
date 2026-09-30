//! f64 references for the Qwen3.5 kernels (`tessl::qwen35`).
//!
//! Each one transcribes `transformers/models/qwen3_5/modeling_qwen3_5.py`, and
//! `tests/qwen35_kernels.rs` holds the two that are easy to get subtly wrong —
//! the delta rule and partial RoPE — to goldens that transformers itself
//! generated (`scripts/gen_qwen35_fixtures.py`). The rest are a few lines each.
//!
//! Pure arithmetic, no GPU. Operands arrive as `f32` because that is what the
//! kernels read; everything after the widening accumulates in `f64`, so the
//! reference is not a meaningful source of the error it measures.

use std::path::{Path, PathBuf};

use tessl::npy::read_npy;

pub const DK: usize = 128;

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// torch's `F.softplus` at its defaults: linear above 20.
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

pub fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Shape of a GDN problem. Key head dim is [`DK`].
#[derive(Clone, Copy, Debug)]
pub struct GdnShape {
    pub b: usize,
    pub t: usize,
    pub hk: usize,
    pub hv: usize,
    pub dv: usize,
}

/// Unpacked GDN operands, transformers' pre-rule tensors:
/// `q`, `k` `[B, T, Hk, 128]`, `v` `[B, T, Hv, Dv]`, raw gate logits `a`, `b`
/// `[B, T, Hv]`, per-head `a_log`, `dt_bias` `[Hv]`, and an optional initial
/// state `[B, Hv, 128, Dv]` (or `[1, ..]` for a shared snapshot).
pub struct GdnProblem<'a> {
    pub s: GdnShape,
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub state0: Option<&'a [f32]>,
    /// `state0` is one state shared by every batch row.
    pub snapshot: bool,
}

/// `x * rsqrt(sum(x^2) + 1e-6)`, transformers' `l2norm`.
fn l2norm(x: &[f32]) -> Vec<f64> {
    let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let inv = 1.0 / (ss + 1e-6).sqrt();
    x.iter().map(|&v| f64::from(v) * inv).collect()
}

/// `Qwen3_5GatedDeltaNet`'s gates and `torch_recurrent_gated_delta_rule` with
/// `use_qk_l2norm_in_kernel=True`, in f64. Returns `(y [B,T,Hv,Dv], state
/// [B,Hv,128,Dv])`.
pub fn gdn_f64(p: &GdnProblem<'_>) -> (Vec<f64>, Vec<f64>) {
    let GdnShape { b, t, hk, hv, dv } = p.s;
    assert_eq!(p.q.len(), b * t * hk * DK, "q");
    assert_eq!(p.k.len(), b * t * hk * DK, "k");
    assert_eq!(p.v.len(), b * t * hv * dv, "v");
    assert_eq!(p.a.len(), b * t * hv, "a");
    assert_eq!(p.b.len(), b * t * hv, "b");
    let group = hv / hk;
    let per = DK * dv;
    let mut state = vec![0.0f64; b * hv * per];
    if let Some(s0) = p.state0 {
        for bi in 0..b {
            for i in 0..hv * per {
                let src = if p.snapshot { i } else { bi * hv * per + i };
                state[bi * hv * per + i] = f64::from(s0[src]);
            }
        }
    }
    let mut y = vec![0.0f64; b * t * hv * dv];
    let scale = (DK as f64).powf(-0.5);
    for bi in 0..b {
        for ti in 0..t {
            for h in 0..hv {
                let kh = h / group;
                let qk_base = ((bi * t + ti) * hk + kh) * DK;
                let q: Vec<f64> = l2norm(&p.q[qk_base..qk_base + DK])
                    .into_iter()
                    .map(|v| v * scale)
                    .collect();
                let k = l2norm(&p.k[qk_base..qk_base + DK]);
                let gi = (bi * t + ti) * hv + h;
                let beta = sigmoid(f64::from(p.b[gi]));
                let g = -f64::from(p.a_log[h]).exp() * softplus(f64::from(p.a[gi]) + f64::from(p.dt_bias[h]));
                let decay = g.exp();
                let s = &mut state[(bi * hv + h) * per..(bi * hv + h + 1) * per];
                for x in s.iter_mut() {
                    *x *= decay;
                }
                let vb = ((bi * t + ti) * hv + h) * dv;
                for n in 0..dv {
                    let kv: f64 = (0..DK).map(|kk| s[kk * dv + n] * k[kk]).sum();
                    let delta = beta * (f64::from(p.v[vb + n]) - kv);
                    for kk in 0..DK {
                        s[kk * dv + n] += k[kk] * delta;
                    }
                    y[vb + n] = (0..DK).map(|kk| s[kk * dv + n] * q[kk]).sum();
                }
            }
        }
    }
    (y, state)
}

/// Depthwise causal conv + SiLU over dense `x [B, T, C]` with `w [C, KW]`,
/// continuing from `state [B|1, C, KW-1]` (zeros when `None`). Returns
/// `(y [B, T, C], new_state [B, C, KW-1])`.
pub fn conv1d_silu_f64(
    x: &[f32],
    w: &[f32],
    state: Option<(&[f32], bool)>,
    b: usize,
    t: usize,
    c: usize,
    kw: usize,
) -> (Vec<f64>, Vec<f64>) {
    let hist = kw - 1;
    let mut y = vec![0.0; b * t * c];
    let mut st = vec![0.0; b * c * hist];
    for bi in 0..b {
        for ci in 0..c {
            let ext = |pos: usize| -> f64 {
                if pos < hist {
                    match state {
                        None => 0.0,
                        Some((s, snapshot)) => {
                            let row = if snapshot { 0 } else { bi };
                            f64::from(s[(row * c + ci) * hist + pos])
                        }
                    }
                } else {
                    f64::from(x[(bi * t + pos - hist) * c + ci])
                }
            };
            for ti in 0..t {
                let acc: f64 = (0..kw).map(|j| f64::from(w[ci * kw + j]) * ext(ti + j)).sum();
                y[(bi * t + ti) * c + ci] = silu(acc);
            }
            for j in 0..hist {
                st[(bi * c + ci) * hist + j] = ext(t + j);
            }
        }
    }
    (y, st)
}

/// `Qwen3_5RMSNormGated`: `rms_norm(x) * w * silu(z)` per `d`-wide head, over
/// dense `[rows * heads, d]` operands.
pub fn gated_rms_norm_f64(x: &[f32], z: &[f32], w: &[f32], d: usize, eps: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for (r, (xr, zr)) in x.chunks(d).zip(z.chunks(d)).enumerate() {
        let ss: f64 = xr.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let inv = 1.0 / (ss / d as f64 + eps).sqrt();
        for i in 0..d {
            out[r * d + i] = f64::from(w[i]) * f64::from(xr[i]) * inv * silu(f64::from(zr[i]));
        }
    }
    out
}

/// One head row through `Qwen3_5RMSNorm` (`* (1 + w)`) and transformers'
/// partial RoPE: pairs `p, p + rotary/2` of the leading `rotary` dims, with
/// `inv_freq = theta^(-2p / rotary)`.
///
/// **The angle is formed in f32**, as transformers forms it
/// (`inv_freq` and `inv_freq @ position_ids` are float32 tensors), and only its
/// cosine and sine are taken in f64. That is not a shortcut: the f32 angle is
/// part of the model's arithmetic. Its rounding is ~6e-8 *relative*, so at
/// position 20000 the angle itself moves by ~1e-3 rad, and an f64-angle
/// reference disagrees with transformers by that much — measured 1.4e-3 on the
/// committed fixture. Matching the model means matching that rounding.
pub fn norm_rope_row_f64(row: &[f32], w: &[f32], rotary: usize, pos: u64, theta: f64, eps: f64) -> Vec<f64> {
    let d = row.len();
    let ss: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let inv = 1.0 / (ss / d as f64 + eps).sqrt();
    let mut x: Vec<f64> = row
        .iter()
        .zip(w)
        .map(|(&v, &wi)| f64::from(v) * inv * (1.0 + f64::from(wi)))
        .collect();
    let half = rotary / 2;
    for p in 0..half {
        let inv_freq = 1.0f32 / (theta as f32).powf((2 * p) as f32 / rotary as f32);
        let angle = f64::from(pos as f32 * inv_freq);
        let (c, s) = (angle.cos(), angle.sin());
        let (x0, x1) = (x[p], x[p + half]);
        x[p] = x0 * c - x1 * s;
        x[p + half] = x1 * c + x0 * s;
    }
    x
}

/// Final norm (`* (w + w_offset)`) and the answer-row logits for one hidden row,
/// with the log-softmax over the answer set.
pub fn score_row_f64(h: &[f32], norm_w: &[f32], w_offset: f64, eps: f64, emb_rows: &[&[f32]]) -> (Vec<f64>, Vec<f64>) {
    let d = h.len();
    let ss: f64 = h.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let inv = 1.0 / (ss / d as f64 + eps).sqrt();
    let logits: Vec<f64> = emb_rows
        .iter()
        .map(|e| {
            (0..d)
                .map(|i| f64::from(h[i]) * inv * (f64::from(norm_w[i]) + w_offset) * f64::from(e[i]))
                .sum()
        })
        .collect();
    let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = mx + logits.iter().map(|l| (l - mx).exp()).sum::<f64>().ln();
    let logp = logits.iter().map(|l| l - lse).collect();
    (logits, logp)
}

// ----------------------------------------------------------------- fixtures ---

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35")
}

/// `(shape, data widened to f64)` of `qwen35_<name>.npy`.
pub fn load(name: &str) -> (Vec<usize>, Vec<f64>) {
    let path = fixture_dir().join(format!("qwen35_{name}.npy"));
    let a = read_npy(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
    let data = match a.f32_slice() {
        Ok(s) => s.iter().map(|&v| f64::from(v)).collect(),
        Err(_) => a.f64_slice().expect("f32 or f64 fixture").to_vec(),
    };
    (a.shape.clone(), data)
}

/// An f32 fixture, as the kernels read it.
pub fn load_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data) = load(name);
    (shape, data.into_iter().map(|v| v as f32).collect())
}
