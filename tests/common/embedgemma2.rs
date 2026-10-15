//! f64 references for EmbeddingGemma 2. `tests/embedgemma2_kernels.rs` holds
//! each one to transformers by fixture (`scripts/gen_embedgemma2_fixtures.py`);
//! [`host_forward`] composes them into a whole host forward, which
//! `tests/embedgemma2_tiny.rs` and `tests/embedgemma2_mm_tiny.rs` hold the
//! device to.

use super::SplitMix;
use tessl::embedgemma2::EmbedGemma2Config;
use tessl::gemm::GemmOperands;
use tessl::safetensors::SafeTensors;
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16};

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

// ---------------------------------------------------------------------------
// The tiny random checkpoint, shared by `tests/embedgemma2_tiny.rs` (which
// holds it to the f64 forward) and `tests/capi_embedgemma2.rs` (which writes it
// to a directory and loads it through the C ABI).
// ---------------------------------------------------------------------------

pub const PREFIX: &str = "language_model.";

/// Three layers: full (d 512, 1 KV head: K/V width 512), sliding (d 256,
/// 1 KV head: 256), sliding (d 256, 2 KV heads: 512).
pub const CONFIG: &str = r#"{"model_type":"embedding_gemma2","text_config":{
"model_type":"embedding_gemma2_text","attention_bias":false,"hidden_activation":"gelu_pytorch_tanh",
"hidden_size":64,"hidden_size_per_layer_input":32,"intermediate_size":128,"vocab_size":48,"embedding_dim":96,
"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"num_hidden_layers":3,"sliding_window":3,
"rms_norm_eps":1e-06,
"layer_types":["full_attention","sliding_attention","sliding_attention"],
"per_layer_config":{"00":{"head_dim":512},"02":{"num_key_value_heads":2}},
"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#;

pub const LAYER_SCALARS: [f32; 3] = [0.75, 1.25, 0.5];

/// A tensor of the tiny checkpoint: name (without the prefix), shape, values.
/// `bf16` tensors are stored as bf16 and kept here already rounded to it.
#[derive(Clone)]
pub struct Weight {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub bf16: bool,
}

pub struct Weights(pub Vec<Weight>);

impl Weights {
    pub fn get(&self, name: &str) -> &[f64] {
        &self
            .0
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("test bug: no weight {name}"))
            .data
    }
}

/// Random weights for `cfg`.
///
/// With `bf16_proj`, every projection (the checkpoint's own dtype for them) is
/// stored as bf16 and kept here rounded to it: a `Bf16` load reads bf16
/// tensors only, and the f64 forward then uses exactly the weights the device
/// holds, so the only error left is the rounding of the activations.
pub fn random_weights(cfg: &EmbedGemma2Config, seed: u64, bf16_proj: bool) -> Weights {
    let mut r = SplitMix::new(seed);
    let mut out = Vec::new();
    let mut push = |name: String, shape: Vec<usize>, scale: f64, offset: f64, bf16: bool| {
        let n: usize = shape.iter().product();
        let mut data: Vec<f64> = (0..n).map(|_| offset + f64::from(r.unit()) * scale).collect();
        if bf16 {
            let bits = f32_slice_to_bf16(&data.iter().map(|&v| v as f32).collect::<Vec<_>>());
            data = bits.iter().map(|&b| f64::from(bf16_bits_to_f32(b))).collect();
        }
        out.push(Weight {
            name,
            shape,
            data,
            bf16,
        });
    };
    let (h, ple, inter, l) = (
        cfg.hidden as usize,
        cfg.ple_dim as usize,
        cfg.intermediate as usize,
        cfg.layers.len(),
    );
    let lin_scale = |fan_in: usize| (fan_in as f64).powf(-0.5);
    push(
        "embed_tokens.weight".into(),
        vec![cfg.vocab as usize, h],
        0.1,
        0.0,
        true,
    );
    push(
        "ple.per_layer_model_projection.weight".into(),
        vec![l * ple, h],
        lin_scale(h),
        0.0,
        bf16_proj,
    );
    push(
        "ple.per_layer_projection_norm.weight".into(),
        vec![ple],
        0.3,
        1.0,
        false,
    );
    push("norm.weight".into(), vec![h], 0.3, 1.0, false);
    push(
        "embedding_projection.weight".into(),
        vec![cfg.embedding_dim as usize, h],
        lin_scale(h),
        0.0,
        bf16_proj,
    );
    for (i, s) in cfg.layers.iter().enumerate() {
        let p = |n: &str| format!("layers.{i}.{n}");
        let (q, kv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            push(p(&format!("{n}.weight")), vec![h], 0.3, 1.0, false);
        }
        push(
            p("self_attn.q_proj.weight"),
            vec![q * d, h],
            lin_scale(h),
            0.0,
            bf16_proj,
        );
        push(
            p("self_attn.k_proj.weight"),
            vec![kv * d, h],
            lin_scale(h),
            0.0,
            bf16_proj,
        );
        push(
            p("self_attn.v_proj.weight"),
            vec![kv * d, h],
            lin_scale(h),
            0.0,
            bf16_proj,
        );
        push(
            p("self_attn.o_proj.weight"),
            vec![h, q * d],
            lin_scale(q * d),
            0.0,
            bf16_proj,
        );
        // Scale 1.0 attention over RMS-normed d-wide heads: weights near
        // d^-0.25 keep q.k O(1), so the softmax spreads over the window
        // instead of collapsing onto one key.
        let qk = (d as f64).powf(-0.25);
        push(p("self_attn.q_norm.weight"), vec![d], 0.3 * qk, qk, false);
        push(p("self_attn.k_norm.weight"), vec![d], 0.3 * qk, qk, false);
        push(p("mlp.gate_proj.weight"), vec![inter, h], lin_scale(h), 0.0, bf16_proj);
        push(p("mlp.up_proj.weight"), vec![inter, h], lin_scale(h), 0.0, bf16_proj);
        push(
            p("mlp.down_proj.weight"),
            vec![h, inter],
            lin_scale(inter),
            0.0,
            bf16_proj,
        );
        push(
            p("ple_block.per_layer_input_gate.weight"),
            vec![ple, h],
            lin_scale(h),
            0.0,
            bf16_proj,
        );
        push(
            p("ple_block.per_layer_projection.weight"),
            vec![h, ple],
            lin_scale(ple),
            0.0,
            bf16_proj,
        );
        push(
            p("ple_block.post_per_layer_input_norm.weight"),
            vec![h],
            0.3,
            1.0,
            false,
        );
        push(p("layer_scalar"), vec![1], 0.0, f64::from(LAYER_SCALARS[i]), false);
    }
    Weights(out)
}

/// safetensors bytes: every weight under `PREFIX`, plus `extra` tensors named
/// as given (f32 zeros of the given shape).
pub fn safetensors_bytes(w: &Weights, extra: &[(&str, &[usize])]) -> Vec<u8> {
    let mut entries: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    for t in &w.0 {
        let bytes: Vec<u8> = if t.bf16 {
            f32_slice_to_bf16(&t.data.iter().map(|&v| v as f32).collect::<Vec<_>>())
                .iter()
                .flat_map(|b| b.to_le_bytes())
                .collect()
        } else {
            t.data.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect()
        };
        entries.push((
            format!("{PREFIX}{}", t.name),
            if t.bf16 { "BF16" } else { "F32" },
            t.shape.clone(),
            bytes,
        ));
    }
    for &(name, shape) in extra {
        let n: usize = shape.iter().product();
        entries.push((name.to_string(), "F32", shape.to_vec(), vec![0u8; n * 4]));
    }
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (i, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            shape.join(","),
            data.len(),
            data.len() + bytes.len()
        ));
        data.extend_from_slice(bytes);
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

// ---------------------------------------------------------------------------
// The whole forward in f64, shared by `tests/embedgemma2_tiny.rs` and
// `tests/embedgemma2_mm_tiny.rs`.
// ---------------------------------------------------------------------------

/// `embed_tokens[id] * sqrt(hidden)` for each id: `inputs_embeds` of a text
/// sequence (transformers: `embed * embed_scale.to(bf16)`; sqrt(64) = 8 is
/// exact).
pub fn embed_ids(cfg: &EmbedGemma2Config, w: &Weights, ids: &[u32]) -> Vec<Vec<f64>> {
    let h = cfg.hidden as usize;
    let table = w.get("embed_tokens.weight");
    let embed_scale = f64::from((h as f32).sqrt());
    ids.iter()
        .map(|&id| {
            table[id as usize * h..(id as usize + 1) * h]
                .iter()
                .map(|v| v * embed_scale)
                .collect()
        })
        .collect()
}

/// `inputs_embeds` of a multimodal sequence, as
/// `EmbeddingGemma2Model.forward` builds them: each id in `placeholders` is
/// gathered as `pad` (through the scaled embedding), then `masked_scatter`
/// writes `soft` (`[n_placeholders, hidden]`, row-major) over those rows in
/// order, unscaled.
pub fn mm_embeds(
    cfg: &EmbedGemma2Config,
    w: &Weights,
    ids: &[u32],
    placeholders: &[u32],
    pad: u32,
    soft: &[f64],
) -> Vec<Vec<f64>> {
    let h = cfg.hidden as usize;
    let gathered: Vec<u32> = ids
        .iter()
        .map(|&id| if placeholders.contains(&id) { pad } else { id })
        .collect();
    let mut embeds = embed_ids(cfg, w, &gathered);
    let mut next = 0;
    for (e, &id) in embeds.iter_mut().zip(ids) {
        if placeholders.contains(&id) {
            e.copy_from_slice(&soft[next * h..(next + 1) * h]);
            next += 1;
        }
    }
    assert_eq!(next * h, soft.len(), "test bug: soft rows and placeholders differ");
    embeds
}

/// The forward of one unpadded text sequence in f64, transformers' order:
/// returns the residual stream after each layer, the final norm's output
/// (each `[T, hidden]`) and the L2-normalized embedding.
pub fn host_forward(cfg: &EmbedGemma2Config, w: &Weights, ids: &[u32]) -> (Vec<Vec<f64>>, Vec<f64>) {
    host_forward_embeds(cfg, w, embed_ids(cfg, w, ids))
}

/// [`host_forward`] from `inputs_embeds` (`language_model(inputs_embeds=...)`):
/// both the residual stream and the per-layer inputs start from `embeds`.
pub fn host_forward_embeds(cfg: &EmbedGemma2Config, w: &Weights, embeds: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (h, ple, inter) = (cfg.hidden as usize, cfg.ple_dim as usize, cfg.intermediate as usize);
    let t = embeds.len();
    let n_layers = cfg.layers.len();
    // Projection-only per-layer inputs: [T, layers, ple].
    let ple_w = w.get("ple.per_layer_model_projection.weight");
    let ple_norm = w.get("ple.per_layer_projection_norm.weight");
    let per_layer: Vec<Vec<Vec<f64>>> = embeds
        .iter()
        .map(|e| {
            let p: Vec<f64> = lin(ple_w, e, n_layers * ple, h)
                .iter()
                .map(|v| v * (h as f64).powf(-0.5))
                .collect();
            (0..n_layers)
                .map(|l| rms(&p[l * ple..(l + 1) * ple], Some(ple_norm)))
                .collect()
        })
        .collect();

    let mut resid = embeds.clone();
    let mut trace = Vec::new();
    for (li, s) in cfg.layers.iter().enumerate() {
        let g = |n: &str| w.get(&format!("layers.{li}.{n}"));
        let (hq, hkv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
        let theta = f64::from(s.rope_theta);
        let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for (pos, r) in resid.iter().enumerate() {
            let x = rms(r, Some(g("input_layernorm.weight")));
            let qp = lin(g("self_attn.q_proj.weight"), &x, hq * d, h);
            let kp = lin(g("self_attn.k_proj.weight"), &x, hkv * d, h);
            let vp = lin(g("self_attn.v_proj.weight"), &x, hkv * d, h);
            for head in 0..hq {
                q.extend(rope(
                    &rms(&qp[head * d..(head + 1) * d], Some(g("self_attn.q_norm.weight"))),
                    pos as f64,
                    theta,
                ));
            }
            for head in 0..hkv {
                k.extend(rope(
                    &rms(&kp[head * d..(head + 1) * d], Some(g("self_attn.k_norm.weight"))),
                    pos as f64,
                    theta,
                ));
                v.extend(rms(&vp[head * d..(head + 1) * d], None));
            }
        }
        let a = Attn {
            b: 1,
            t,
            h: hq,
            hkv,
            d,
            window: if s.sliding { cfg.sliding_window as usize } else { 0 },
            lens: vec![t],
        };
        let o = attn_ref(&a, &q, &k, &v, None);
        for (pos, r) in resid.iter_mut().enumerate() {
            let y = lin(
                g("self_attn.o_proj.weight"),
                &o[pos * hq * d..(pos + 1) * hq * d],
                h,
                hq * d,
            );
            let y = rms(&y, Some(g("post_attention_layernorm.weight")));
            r.iter_mut().zip(&y).for_each(|(a, b)| *a += b);

            let x = rms(r, Some(g("pre_feedforward_layernorm.weight")));
            let gate = lin(g("mlp.gate_proj.weight"), &x, inter, h);
            let up = lin(g("mlp.up_proj.weight"), &x, inter, h);
            let mid: Vec<f64> = gate.iter().zip(&up).map(|(a, b)| gelu_tanh(*a) * b).collect();
            let y = rms(
                &lin(g("mlp.down_proj.weight"), &mid, h, inter),
                Some(g("post_feedforward_layernorm.weight")),
            );
            r.iter_mut().zip(&y).for_each(|(a, b)| *a += b);

            let gp = lin(g("ple_block.per_layer_input_gate.weight"), r, ple, h);
            let gm: Vec<f64> = gp
                .iter()
                .zip(&per_layer[pos][li])
                .map(|(a, b)| gelu_tanh(*a) * b)
                .collect();
            let y = rms(
                &lin(g("ple_block.per_layer_projection.weight"), &gm, h, ple),
                Some(g("ple_block.post_per_layer_input_norm.weight")),
            );
            let scalar = g("layer_scalar")[0];
            r.iter_mut().zip(&y).for_each(|(a, b)| *a = (*a + b) * scalar);
        }
        trace.push(resid.concat());
    }
    let normed: Vec<Vec<f64>> = resid.iter().map(|r| rms(r, Some(w.get("norm.weight")))).collect();
    trace.push(normed.concat());
    // Project every token, then mean-pool (transformers' order; the Rust side
    // pools first, which is the same linear map).
    let proj = w.get("embedding_projection.weight");
    let e = cfg.embedding_dim as usize;
    let mut mean = vec![0.0; e];
    for n in &normed {
        lin(proj, n, e, h)
            .iter()
            .zip(mean.iter_mut())
            .for_each(|(p, m)| *m += p / t as f64);
    }
    (trace, l2_normalize(&mean))
}
/// What a mode is held to against the f64 forward.
#[derive(Clone, Copy)]
pub struct Bounds {
    /// Each layer's residual and the final norm, of their largest magnitude.
    pub layer_rel: f64,
    /// Embeddings, max abs.
    pub emb_abs: f64,
    /// Embeddings, cosine at least.
    pub emb_cos: f64,
}

pub fn bounds(operands: GemmOperands) -> Bounds {
    match operands {
        GemmOperands::ExactF32 => Bounds {
            layer_rel: 1e-4,
            emb_abs: 1e-4,
            emb_cos: 0.99999,
        },
        GemmOperands::Bf16 => Bounds {
            layer_rel: 5e-2,
            emb_abs: 1e-2,
            emb_cos: 0.9999,
        },
    }
}

/// The tiny checkpoint's weights for a mode: the `Bf16` load reads bf16
/// projections only.
pub fn weights_for(cfg: &EmbedGemma2Config, seed: u64, operands: GemmOperands) -> Weights {
    random_weights(cfg, seed, operands == GemmOperands::Bf16)
}

pub fn open(tag: &str, bytes: Vec<u8>) -> SafeTensors {
    SafeTensors::from_bytes(tag, bytes).unwrap()
}

pub fn max_abs(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - y).abs())
        .fold(0.0, f64::max)
}

pub fn cosine(a: &[f32], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * y).sum();
    let na = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    dot / (na * nb)
}
