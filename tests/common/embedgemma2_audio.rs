//! f64 references for EmbeddingGemma 2's audio tower, written the way
//! transformers 5.19's `Gemma4AudioModel` is: the chunked attention builds the
//! blocked q/k/v, the relative-position product and `_rel_shift`
//! (pad, flatten, truncate, view) literally, and the mask is the 4-D mask
//! converted to the 5-D blocked layout by `_convert_4d_mask_to_blocked_5d`'s
//! pad/gather. None of `tessl::embedgemma2_audio`'s index helpers are used, so
//! the kernels and the host code that mirrors them are held to an independent
//! derivation. `tests/embedgemma2_audio_tiny.rs` composes them into a whole host
//! forward; `tests/embedgemma2_audio_kernels.rs` holds each kernel to its piece.
//!
//! The mask is the intended (bool, sdpa) one, not the inverted float mask that
//! transformers' eager path applies: see `src/embedgemma2_audio.rs`.

use std::collections::HashMap;

use tessl::embedgemma2_audio::AudioConfig;

/// `torch.clamp`: `min(max(x, lo), hi)`.
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// `F.softplus` (threshold 20).
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `x * (mean(x^2) + eps)^-0.5 * w`, or no weight.
pub fn rms_eps(x: &[f64], w: Option<&[f64]>, eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = (ms + eps).powf(-0.5);
    x.iter()
        .enumerate()
        .map(|(i, v)| v * inv * w.map_or(1.0, |w| w[i]))
        .collect()
}

/// One subsampling layer on `x: [cin][t][f]` (channels first, as torch holds
/// it): the frames are multiplied by `mask` (`[t]`), a 3x3 stride-2 pad-1
/// convolution with `w: [cout][cin][3][3]` and no bias, `LayerNorm` over the
/// channels with `gamma` and no bias (biased variance), ReLU. Returns
/// `[cout][t'][f']` and the mask `mask[::2]`.
pub fn conv_layer(
    x: &[f64],
    (cin, t, f): (usize, usize, usize),
    mask: &[bool],
    w: &[f64],
    gamma: &[f64],
    eps: f64,
) -> (Vec<f64>, (usize, usize, usize), Vec<bool>) {
    let cout = gamma.len();
    let (to, fo) = (t.div_ceil(2), f.div_ceil(2));
    let at = |c: usize, r: i64, col: i64| -> f64 {
        if r < 0 || col < 0 || r >= t as i64 || col >= f as i64 || !mask[r as usize] {
            0.0
        } else {
            x[(c * t + r as usize) * f + col as usize]
        }
    };
    let mut conv = vec![0.0; cout * to * fo];
    for co in 0..cout {
        for ti in 0..to {
            for fi in 0..fo {
                let mut s = 0.0;
                for ci in 0..cin {
                    for kh in 0..3 {
                        for kw in 0..3 {
                            s += w[((co * cin + ci) * 3 + kh) * 3 + kw]
                                * at(ci, (2 * ti + kh) as i64 - 1, (2 * fi + kw) as i64 - 1);
                        }
                    }
                }
                conv[(co * to + ti) * fo + fi] = s;
            }
        }
    }
    let mut out = vec![0.0; cout * to * fo];
    for ti in 0..to {
        for fi in 0..fo {
            let col: Vec<f64> = (0..cout).map(|c| conv[(c * to + ti) * fo + fi]).collect();
            let mean = col.iter().sum::<f64>() / cout as f64;
            let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / cout as f64;
            for c in 0..cout {
                out[(c * to + ti) * fo + fi] = ((col[c] - mean) / (var + eps).sqrt() * gamma[c]).max(0.0);
            }
        }
    }
    (out, (cout, to, fo), mask.iter().step_by(2).copied().collect())
}

/// `Gemma4AudioRelPositionalEncoding`: `[ctx / 2 + 1, hidden]`, row `p` at
/// position `ctx / 2 - p`.
pub fn pos_table(hidden: usize, ctx: usize) -> Vec<f64> {
    let n = hidden / 2;
    let inc = 10000f64.ln() / (n.max(2) - 1) as f64;
    let mut out = Vec::new();
    for pos in (0..=ctx / 2).rev() {
        let t: Vec<f64> = (0..n).map(|d| pos as f64 * (-(d as f64) * inc).exp()).collect();
        out.extend(t.iter().map(|v| v.sin()));
        out.extend(t.iter().map(|v| v.cos()));
    }
    out
}

/// `relative_k_proj(pos_table)`: `[ctx / 2 + 1, hidden]`.
pub fn relative_keys(hidden: usize, ctx: usize, w: &[f64]) -> Vec<f64> {
    let table = pos_table(hidden, ctx);
    let rows = ctx / 2 + 1;
    let mut out = vec![0.0; rows * hidden];
    for p in 0..rows {
        for n in 0..hidden {
            out[p * hidden + n] = (0..hidden).map(|k| table[p * hidden + k] * w[n * hidden + k]).sum();
        }
    }
    out
}

/// `head_dim^-0.5 / ln 2 * softplus(per_dim_scale)`.
pub fn q_mult(per_dim: &[f64]) -> Vec<f64> {
    let q_scale = (per_dim.len() as f64).powf(-0.5) / 2f64.ln();
    per_dim.iter().map(|&s| q_scale * softplus(s)).collect()
}

/// The shape and constants of the attention.
#[derive(Clone, Copy)]
pub struct AttnSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub chunk: usize,
    /// `attention_context_left`.
    pub left: usize,
    pub cap: f64,
    pub invalid: f64,
}

/// `Gemma4AudioAttention.forward` after the projections: `q`, `k`, `v` are
/// `[t][heads * head_dim]` (output-clamped), `rel` is `relative_keys`, `omask`
/// the tower mask. Returns `[t][heads * head_dim]` before the output
/// projection. Literal: blocked q, context-window k and v, `matrix_ac`,
/// `matrix_bd`, `_rel_shift`, the 5-D mask, softcap, masked fill, softmax.
pub fn attention_ref(
    s: &AttnSpec,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    rel: &[f64],
    q_mult: &[f64],
    omask: &[bool],
) -> Vec<f64> {
    let (nh, hd, chunk) = (s.heads, s.head_dim, s.chunk);
    let hidden = nh * hd;
    let past = s.left - 1;
    let ctx = chunk + past;
    let t = omask.len();
    let blocks = t.div_ceil(chunk);
    let rel_rows = rel.len() / hidden;
    let k_scale = (1.0 + 1f64.exp()).ln() / 2f64.ln();

    // _convert_to_block: queries (scaled), zero padded to whole blocks.
    let qb = |b: usize, i: usize, h: usize, d: usize| -> f64 {
        let r = b * chunk + i;
        if r < t {
            q[r * hidden + h * hd + d] * q_mult[d]
        } else {
            0.0
        }
    };
    // _extract_block_context: padded by `past` on the left and enough on the
    // right; block b's context slot j is padded position b * chunk + j.
    let ctx_at = |x: &[f64], scale: f64, b: usize, j: usize, h: usize, d: usize| -> f64 {
        let p = b * chunk + j;
        if p < past || p - past >= t {
            0.0
        } else {
            x[(p - past) * hidden + h * hd + d] * scale
        }
    };

    // The 4-D mask [t, t] (window function and key padding), converted to the
    // blocked 5-D layout [blocks, chunk, ctx] as _convert_4d_mask_to_blocked_5d.
    let padded = blocks * chunk;
    let m4 = |qi: usize, kv: usize| -> bool {
        if qi >= t || kv >= t {
            return false;
        }
        let dist = qi as i64 - kv as i64;
        (0..past as i64).contains(&dist) && omask[kv]
    };
    let m5 = |b: usize, i: usize, j: usize| -> bool {
        let col = b * chunk + j;
        col >= past && col - past < padded && m4(b * chunk + i, col - past)
    };

    let mut out = vec![0.0; t * hidden];
    for h in 0..nh {
        for b in 0..blocks {
            // matrix_bd before the shift: [chunk, rel_rows], then _rel_shift.
            let mut bd = vec![0.0; chunk * rel_rows];
            for i in 0..chunk {
                for p in 0..rel_rows {
                    bd[i * rel_rows + p] = (0..hd).map(|d| qb(b, i, h, d) * rel[p * hidden + h * hd + d]).sum();
                }
            }
            // pad each row to ctx + 1, flatten, keep chunk * ctx, view [chunk, ctx]
            let mut flat = Vec::with_capacity(chunk * (ctx + 1));
            for i in 0..chunk {
                for c in 0..=ctx {
                    flat.push(if c < rel_rows { bd[i * rel_rows + c] } else { 0.0 });
                }
            }
            flat.truncate(chunk * ctx);
            for i in 0..chunk {
                if b * chunk + i >= t {
                    continue;
                }
                let mut logits: Vec<f64> = (0..ctx)
                    .map(|j| {
                        let ac: f64 = (0..hd).map(|d| qb(b, i, h, d) * ctx_at(k, k_scale, b, j, h, d)).sum();
                        let x = (ac + flat[i * ctx + j]) / s.cap;
                        let x = x.tanh() * s.cap;
                        if m5(b, i, j) {
                            x
                        } else {
                            s.invalid
                        }
                    })
                    .collect();
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                logits.iter_mut().for_each(|l| *l = (*l - m).exp());
                let z: f64 = logits.iter().sum();
                for d in 0..hd {
                    out[(b * chunk + i) * hidden + h * hd + d] =
                        (0..ctx).map(|j| logits[j] / z * ctx_at(v, 1.0, b, j, h, d)).sum();
                }
            }
        }
    }
    out
}

/// Weights by their checkpoint names (`audio_tower.layers.0...`), in f64.
pub type Weights = HashMap<String, Vec<f64>>;

/// A `Gemma4ClippableLinear` on one row: clamp in, `w [out, in]`, clamp out.
pub fn clip_lin(w: &Weights, name: &str, x: &[f64], out_f: usize) -> Vec<f64> {
    let bound = |s: &str| w[&format!("{name}.{s}")][0];
    let (imin, imax, omin, omax) = (
        bound("input_min"),
        bound("input_max"),
        bound("output_min"),
        bound("output_max"),
    );
    let wt = &w[&format!("{name}.linear.weight")];
    let in_f = x.len();
    let xc: Vec<f64> = x.iter().map(|&v| clamp(v, imin, imax)).collect();
    (0..out_f)
        .map(|o| clamp((0..in_f).map(|i| wt[o * in_f + i] * xc[i]).sum(), omin, omax))
        .collect()
}

/// The whole tower and embedder on one clip. Intermediates are `[t][width]`.
pub struct AudioRef {
    pub subsample: Vec<f64>,
    pub layers: Vec<Vec<f64>>,
    pub tower: Vec<f64>,
    pub out_mask: Vec<bool>,
    /// The kept rows of the embedder's output, `[kept][embed_dim]`.
    pub soft: Vec<f64>,
}

pub fn audio_forward(cfg: &AudioConfig, w: &Weights, feats: &[f64], mask: &[bool]) -> AudioRef {
    let h = cfg.hidden as usize;
    let eps = f64::from(cfg.eps);
    let gc = f64::from(cfg.gradient_clip);
    let mels = cfg.mel_bins() as usize;
    let frames = mask.len();
    let p = |s: &str| &w[&format!("audio_tower.{s}")];
    let sc = |s: &str| format!("subsample_conv_projection.{s}");

    let (x, d0, m1) = conv_layer(
        feats,
        (1, frames, mels),
        mask,
        p(&sc("layer0.conv.weight")),
        p(&sc("layer0.norm.weight")),
        eps,
    );
    let (x, d1, om) = conv_layer(
        &x,
        d0,
        &m1,
        p(&sc("layer1.conv.weight")),
        p(&sc("layer1.norm.weight")),
        eps,
    );
    let (c1, t, f2) = d1;
    // permute(0, 2, 3, 1).reshape(t, -1): row t holds [f2][c1].
    let proj = p(&sc("input_proj_linear.weight"));
    let proj_in = f2 * c1;
    assert_eq!(proj.len(), h * proj_in, "test bug: input_proj shape");
    let mut hs: Vec<Vec<f64>> = (0..t)
        .map(|ti| {
            let row: Vec<f64> = (0..f2)
                .flat_map(|fi| (0..c1).map(move |c| (c, fi)))
                .map(|(c, fi)| x[(c * t + ti) * f2 + fi])
                .collect();
            (0..h)
                .map(|o| (0..proj_in).map(|i| proj[o * proj_in + i] * row[i]).sum())
                .collect()
        })
        .collect();
    let subsample = hs.concat();

    let hd = h / cfg.heads as usize;
    let spec = AttnSpec {
        heads: cfg.heads as usize,
        head_dim: hd,
        chunk: cfg.chunk as usize,
        left: cfg.context_left as usize,
        cap: f64::from(cfg.logit_cap),
        invalid: f64::from(cfg.invalid_logit),
    };
    let ctx = cfg.ctx() as usize;
    let rw = f64::from(cfg.residual_weight);
    let mut layers = Vec::new();
    for li in 0..cfg.layers as usize {
        let lp = |s: &str| format!("layers.{li}.{s}");
        let cl = |v: &[f64]| -> Vec<f64> { v.iter().map(|&a| clamp(a, -gc, gc)).collect() };
        let ff = |hs: &mut Vec<Vec<f64>>, name: &str| {
            for r in hs.iter_mut() {
                let a = rms_eps(&cl(r), Some(p(&lp(&format!("{name}.pre_layer_norm.weight")))), eps);
                let a = clip_lin(
                    w,
                    &format!("audio_tower.{}", lp(&format!("{name}.ffw_layer_1"))),
                    &a,
                    4 * h,
                );
                let a: Vec<f64> = a.iter().map(|&v| silu(v)).collect();
                let a = clip_lin(w, &format!("audio_tower.{}", lp(&format!("{name}.ffw_layer_2"))), &a, h);
                let a = rms_eps(&cl(&a), Some(p(&lp(&format!("{name}.post_layer_norm.weight")))), eps);
                r.iter_mut().zip(&a).for_each(|(x, y)| *x += y * rw);
            }
        };
        ff(&mut hs, "feed_forward1");

        // Attention.
        let lin = |name: &str, x: &[f64], out_f: usize| clip_lin(w, &format!("audio_tower.{}", lp(name)), x, out_f);
        let normed: Vec<Vec<f64>> = hs
            .iter()
            .map(|r| rms_eps(&cl(r), Some(p(&lp("norm_pre_attn.weight"))), eps))
            .collect();
        let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for n in &normed {
            q.extend(lin("self_attn.q_proj", n, h));
            k.extend(lin("self_attn.k_proj", n, h));
            v.extend(lin("self_attn.v_proj", n, h));
        }
        let rel = relative_keys(h, ctx, p(&lp("self_attn.relative_k_proj.weight")));
        let qm = q_mult(p(&lp("self_attn.per_dim_scale")));
        let o = attention_ref(&spec, &q, &k, &v, &rel, &qm, &om);
        for (ti, r) in hs.iter_mut().enumerate() {
            let a = lin("self_attn.post", &o[ti * h..(ti + 1) * h], h);
            let a = rms_eps(&cl(&a), Some(p(&lp("norm_post_attn.weight"))), eps);
            r.iter_mut().zip(&a).for_each(|(x, y)| *x += y);
        }

        // Light conv: pre-norm, linear_start, glu, causal depthwise conv.
        let mut glus: Vec<Vec<f64>> = Vec::new();
        for r in &hs {
            let a = rms_eps(r, Some(p(&lp("lconv1d.pre_layer_norm.weight"))), eps);
            let a = lin("lconv1d.linear_start", &a, 2 * h);
            glus.push((0..h).map(|c| a[c] * sigmoid(a[h + c])).collect());
        }
        let taps = cfg.conv_kernel as usize;
        let cw = p(&lp("lconv1d.depthwise_conv1d.weight"));
        for (ti, r) in hs.iter_mut().enumerate() {
            let conv: Vec<f64> = (0..h)
                .map(|c| {
                    (0..taps)
                        .map(|j| {
                            let s = ti as i64 - (taps as i64 - 1) + j as i64;
                            if s < 0 {
                                0.0
                            } else {
                                cw[c * taps + j] * glus[s as usize][c]
                            }
                        })
                        .sum()
                })
                .collect();
            let a = rms_eps(&cl(&conv), Some(p(&lp("lconv1d.conv_norm.weight"))), eps);
            let a: Vec<f64> = a.iter().map(|&v| silu(v)).collect();
            let a = lin("lconv1d.linear_end", &a, h);
            r.iter_mut().zip(&a).for_each(|(x, y)| *x += y);
        }

        ff(&mut hs, "feed_forward2");
        for r in hs.iter_mut() {
            *r = rms_eps(&cl(r), Some(p(&lp("norm_out.weight"))), eps);
        }
        layers.push(hs.concat());
    }

    let ow = p("output_proj.weight");
    let ob = p("output_proj.bias");
    let od = cfg.out_dims as usize;
    let tower_rows: Vec<Vec<f64>> = hs
        .iter()
        .map(|r| {
            (0..od)
                .map(|o| (0..h).map(|i| ow[o * h + i] * r[i]).sum::<f64>() + ob[o])
                .collect()
        })
        .collect();
    let ew = &w["embed_audio.embedding_projection.weight"];
    let ed = cfg.embed_dim as usize;
    let mut soft = Vec::new();
    for (ti, r) in tower_rows.iter().enumerate() {
        if om[ti] {
            let n = rms_eps(r, None, eps);
            soft.extend((0..ed).map(|o| (0..od).map(|i| ew[o * od + i] * n[i]).sum::<f64>()));
        }
    }
    AudioRef {
        subsample,
        layers,
        tower: tower_rows.concat(),
        out_mask: om,
        soft,
    }
}
