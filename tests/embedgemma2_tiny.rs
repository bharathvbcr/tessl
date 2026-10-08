//! The whole EmbeddingGemma 2 forward (`EmbedGemma2Model::load` + `encode`)
//! on a tiny random checkpoint, against an f64 host forward composed from the
//! references `tests/embedgemma2_kernels.rs` holds to transformers
//! (`tests/common/embedgemma2.rs`). It runs everywhere `cargo test` reaches a
//! GPU, with no checkpoint: the real one is `tests/embedgemma2_model.rs`.
//!
//! What it pins that the per-kernel tests cannot: layer order, the
//! `sqrt(hidden)` embedding scale, the per-layer input slice each layer takes,
//! the layer scalar, the final norm, the projection's orientation, pooling
//! over each sequence's own rows, and the K/V layout of layers whose
//! `kv_heads * head_dim` differ (the checkpoint's layers all have 512, so it
//! cannot). The config puts a narrower layer after a wider one, with a batch
//! of four, a window of 3 below most lengths, and a length-1 sequence.
//!
//! The checkpoint is built in memory as safetensors bytes and opened with
//! `SafeTensors::from_bytes`; nothing touches the file system.
//!
//! Bounds, fixed before any run (the same as the real-checkpoint test): each
//! layer's residual stream and the final norm within `1e-4` of their largest
//! magnitude; embeddings within `1e-4` max abs and cosine `>= 0.99999`; each
//! sequence alone vs inside the batch within `1e-5`.

mod common;

use common::embedgemma2::{attn_ref, gelu_tanh, l2_normalize, lin, rms, rope, Attn};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model};
use tessl::infer_trace;
use tessl::safetensors::SafeTensors;
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16};

const PREFIX: &str = "language_model.";
const LAYER_REL: f64 = 1e-4;
const EMB_ABS: f64 = 1e-4;
const EMB_COS: f64 = 0.99999;
const BATCH_ABS: f64 = 1e-5;

/// Three layers: full (d 512, 1 KV head: K/V width 512), sliding (d 256,
/// 1 KV head: 256), sliding (d 256, 2 KV heads: 512).
const CONFIG: &str = r#"{"model_type":"embedding_gemma2","text_config":{
"model_type":"embedding_gemma2_text","attention_bias":false,"hidden_activation":"gelu_pytorch_tanh",
"hidden_size":64,"hidden_size_per_layer_input":32,"intermediate_size":128,"vocab_size":48,"embedding_dim":96,
"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"num_hidden_layers":3,"sliding_window":3,
"rms_norm_eps":1e-06,
"layer_types":["full_attention","sliding_attention","sliding_attention"],
"per_layer_config":{"00":{"head_dim":512},"02":{"num_key_value_heads":2}},
"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#;

const LAYER_SCALARS: [f32; 3] = [0.75, 1.25, 0.5];

/// A tensor of the tiny checkpoint: name (without the prefix), shape, values.
/// `bf16` tensors are stored as bf16 and kept here already rounded to it.
#[derive(Clone)]
struct Weight {
    name: String,
    shape: Vec<usize>,
    data: Vec<f64>,
    bf16: bool,
}

struct Weights(Vec<Weight>);

impl Weights {
    fn get(&self, name: &str) -> &[f64] {
        &self
            .0
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("test bug: no weight {name}"))
            .data
    }
}

fn random_weights(cfg: &EmbedGemma2Config, seed: u64) -> Weights {
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
        false,
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
        false,
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
        push(p("self_attn.q_proj.weight"), vec![q * d, h], lin_scale(h), 0.0, false);
        push(p("self_attn.k_proj.weight"), vec![kv * d, h], lin_scale(h), 0.0, false);
        push(p("self_attn.v_proj.weight"), vec![kv * d, h], lin_scale(h), 0.0, false);
        push(
            p("self_attn.o_proj.weight"),
            vec![h, q * d],
            lin_scale(q * d),
            0.0,
            false,
        );
        // Scale 1.0 attention over RMS-normed d-wide heads: weights near
        // d^-0.25 keep q.k O(1), so the softmax spreads over the window
        // instead of collapsing onto one key.
        let qk = (d as f64).powf(-0.25);
        push(p("self_attn.q_norm.weight"), vec![d], 0.3 * qk, qk, false);
        push(p("self_attn.k_norm.weight"), vec![d], 0.3 * qk, qk, false);
        push(p("mlp.gate_proj.weight"), vec![inter, h], lin_scale(h), 0.0, false);
        push(p("mlp.up_proj.weight"), vec![inter, h], lin_scale(h), 0.0, false);
        push(p("mlp.down_proj.weight"), vec![h, inter], lin_scale(inter), 0.0, false);
        push(
            p("ple_block.per_layer_input_gate.weight"),
            vec![ple, h],
            lin_scale(h),
            0.0,
            false,
        );
        push(
            p("ple_block.per_layer_projection.weight"),
            vec![h, ple],
            lin_scale(ple),
            0.0,
            false,
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
fn safetensors_bytes(w: &Weights, extra: &[(&str, &[usize])]) -> Vec<u8> {
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

fn open(tag: &str, bytes: Vec<u8>) -> SafeTensors {
    SafeTensors::from_bytes(tag, bytes).unwrap()
}

/// The forward of one unpadded sequence in f64, transformers' order:
/// returns the residual stream after each layer, the final norm's output
/// (each `[T, hidden]`) and the L2-normalized embedding.
fn host_forward(cfg: &EmbedGemma2Config, w: &Weights, ids: &[u32]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (h, ple, inter) = (cfg.hidden as usize, cfg.ple_dim as usize, cfg.intermediate as usize);
    let t = ids.len();
    let n_layers = cfg.layers.len();
    let table = w.get("embed_tokens.weight");
    // transformers: embed * embed_scale.to(bf16); sqrt(64) = 8 is exact.
    let embed_scale = f64::from((h as f32).sqrt());
    let embeds: Vec<Vec<f64>> = ids
        .iter()
        .map(|&id| {
            table[id as usize * h..(id as usize + 1) * h]
                .iter()
                .map(|v| v * embed_scale)
                .collect()
        })
        .collect();
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

fn max_abs(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - y).abs())
        .fold(0.0, f64::max)
}

fn cosine(a: &[f32], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * y).sum();
    let na = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn sequences(vocab: u32) -> Vec<Vec<u32>> {
    let mut r = SplitMix::new(77);
    [9usize, 4, 1, 7]
        .iter()
        .map(|&n| (0..n).map(|_| r.range(0, vocab as usize - 1) as u32).collect())
        .collect()
}

#[test]
fn tiny_checkpoint_matches_the_f64_forward() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let widths: Vec<u32> = cfg.layers.iter().map(|s| s.kv_heads * s.head_dim).collect();
    assert_eq!(
        widths,
        [512, 256, 512],
        "test bug: the config must mix K/V widths, narrow after wide"
    );
    let w = random_weights(&cfg, 2026);
    let st = open("ok", safetensors_bytes(&w, &[("vision_tower.unused.weight", &[3])]));
    let seqs = sequences(cfg.vocab);
    let (h, dim) = (cfg.hidden as usize, cfg.embedding_dim as usize);

    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();

        // 1. Layer by layer, the longest sequence (past the window both ways).
        let (want_trace, _) = host_forward(&cfg, &w, &seqs[0]);
        let out = model.encode(&[&seqs[0]], None, true).unwrap();
        assert_eq!(out.trace.len(), cfg.layers.len() + 1);
        for (i, (got, want)) in out.trace.iter().zip(&want_trace).enumerate() {
            let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let err = max_abs(&got[..seqs[0].len() * h], want);
            eprintln!("trace {i}: max abs {err:.3e} of scale {scale:.3e}");
            assert!(
                err <= LAYER_REL * scale,
                "trace {i}: max abs {err:.3e} > {LAYER_REL:.0e} * {scale:.3e}"
            );
        }

        // 2. Every sequence alone, against the f64 forward.
        let mut alone = Vec::new();
        for (n, ids) in seqs.iter().enumerate() {
            let (_, want) = host_forward(&cfg, &w, ids);
            let e = model.encode(&[ids], None, false).unwrap().embeddings;
            let (err, cos) = (max_abs(&e, &want), cosine(&e, &want));
            eprintln!(
                "sequence {n} ({} tokens) alone: max abs {err:.3e}, cosine {cos:.9}",
                ids.len()
            );
            assert!(
                err <= EMB_ABS && cos >= EMB_COS,
                "sequence {n}: max abs {err:.3e}, cosine {cos:.9}"
            );
            alone.push((e, want));
        }

        // 3. All four in one forward: rows b > 0 of every layer's K/V.
        let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        let out = model.encode(&refs, None, false).unwrap();
        assert_eq!(out.forwards, 1, "the tiny batch must share one forward");
        for (n, (e, want)) in alone.iter().enumerate() {
            let got = &out.embeddings[n * dim..(n + 1) * dim];
            let (err, cos) = (max_abs(got, want), cosine(got, want));
            let drift = got.iter().zip(e).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("sequence {n} batched: max abs {err:.3e}, cosine {cos:.9}, vs alone {drift:.3e}");
            assert!(
                err <= EMB_ABS && cos >= EMB_COS,
                "sequence {n} batched: max abs {err:.3e}, cosine {cos:.9}"
            );
            assert!(
                f64::from(drift) <= BATCH_ABS,
                "sequence {n}: batched vs alone {drift:.3e}"
            );
        }

        // 4. Matryoshka prefixes from the GPU: the f64 forward's prefix,
        //    renormalized, batched like (3).
        for d in [64usize, 17, 1] {
            let out = model.encode(&refs, Some(d as u32), false).unwrap();
            assert_eq!(out.embeddings.len(), seqs.len() * d);
            for (n, (_, want)) in alone.iter().enumerate() {
                let want = l2_normalize(&want[..d]);
                let got = &out.embeddings[n * d..(n + 1) * d];
                let (err, cos) = (max_abs(got, &want), cosine(got, &want));
                assert!(
                    err <= EMB_ABS && cos >= EMB_COS,
                    "prefix {d}, sequence {n}: max abs {err:.3e}, cosine {cos:.9}"
                );
            }
        }
        assert!(model.encode(&refs, Some(0), false).is_err(), "truncate_dim 0");
        assert!(
            model.encode(&refs, Some(cfg.embedding_dim + 1), false).is_err(),
            "truncate_dim past embedding_dim"
        );
    });
}

/// What `f` counted.
fn traced<T>(f: impl FnOnce() -> T) -> (T, infer_trace::Snapshot) {
    infer_trace::set_enabled(true);
    let s0 = infer_trace::snapshot();
    let out = f();
    let s1 = infer_trace::snapshot();
    infer_trace::set_enabled(false);
    (out, s1.since(&s0))
}

/// An encode's forwards share one set of activations, none of it zeroed on
/// the host, and poisoning every unzeroed allocation with NaN changes no bit
/// of any embedding: no kernel reads an element its forward did not write.
/// Before the activations were reused, each forward allocated its own (the
/// two-forward encode made twice the one-forward encode's allocations) and
/// zeroed them on the host.
#[test]
fn encode_reuses_unzeroed_activations() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 7);
    let st = open("reuse", safetensors_bytes(&w, &[]));
    // A long sequence beside short ones: `encode` runs the long one alone and
    // the short ones together in a second, smaller forward.
    let long: Vec<u32> = (0..4000u32).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let short: Vec<Vec<u32>> = (0..7u32)
        .map(|s| (0..30u32).map(|i| (i * 5 + s) % cfg.vocab).collect())
        .collect();
    let mut batch: Vec<&[u32]> = vec![&long];
    batch.extend(short.iter().map(Vec::as_slice));
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        let (one, t1) = traced(|| model.encode(&batch[1..2], None, false).unwrap());
        let (two, t2) = traced(|| model.encode(&batch, None, false).unwrap());
        assert_eq!((one.forwards, two.forwards), (1, 2), "test bug: the split changed");
        assert_eq!(
            t1.host_zero_bytes, 0,
            "one forward zeroed {} bytes on the host",
            t1.host_zero_bytes
        );
        assert_eq!(
            t2.host_zero_bytes, 0,
            "two forwards zeroed {} bytes on the host",
            t2.host_zero_bytes
        );
        assert_eq!(
            t2.cold_allocs, t1.cold_allocs,
            "two forwards made {} allocations, one made {}: the activations are not shared",
            t2.cold_allocs, t1.cold_allocs
        );

        rt.set_poison_unzeroed(true);
        let poisoned = model.encode(&batch, None, false);
        rt.set_poison_unzeroed(false);
        let poisoned = poisoned.unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&poisoned.embeddings),
            bits(&two.embeddings),
            "an unzeroed activation was read before it was written"
        );
        assert!(two.embeddings.iter().all(|v| v.is_finite()));

        // The second forward runs in buffers sized for the first: its K/V
        // stride is 4000 / 7 = 571 positions (1142 in the 256-wide layer),
        // not its own 30. Each sequence must still match itself
        // run alone in exactly sized buffers.
        let dim = cfg.embedding_dim as usize;
        for (n, ids) in batch.iter().enumerate() {
            let alone = model.encode(&[ids], None, false).unwrap().embeddings;
            let got = &two.embeddings[n * dim..(n + 1) * dim];
            let drift = got
                .iter()
                .zip(&alone)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                f64::from(drift) <= BATCH_ABS,
                "sequence {n} ({} tokens) in the two-forward batch vs alone: {drift:.3e}",
                ids.len()
            );
        }
    });
}

/// Weights are long-lived (`BufferKind::Hot`): dropping the model hands their
/// memory back to the device. Allocated as mid-step temporaries (`Cold`) they
/// would park in the runtime's freelist instead, and the device would still
/// hold them after the model is gone.
#[test]
fn dropping_the_model_releases_its_weights() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 2026);
    let st = open("drop", safetensors_bytes(&w, &[]));
    with_gpu(|rt| {
        rt.synchronize().unwrap();
        let before = rt.current_allocated_bytes();
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        let loaded = rt.current_allocated_bytes();
        let weights: u64 =
            w.0.iter()
                .map(|t| t.data.len() as u64 * if t.bf16 { 2 } else { 4 })
                .sum();
        // Buffers under 16 KiB (the norms) are not charged one by one, so the
        // load is held to half its bytes: enough to show the weights are seen.
        assert!(
            loaded >= before + weights / 2,
            "loading added {} bytes for {weights} of weights",
            loaded - before
        );
        drop(model);
        rt.synchronize().unwrap();
        let after = rt.current_allocated_bytes();
        assert!(
            after <= before,
            "{} bytes still allocated after the model dropped",
            after.saturating_sub(before)
        );
    });
}

#[test]
fn loader_refuses_tensors_it_would_ignore() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 2026);
    with_gpu(|rt| {
        for extra in [
            "language_model.layers.1.self_attn.q_proj.bias",
            "language_model.norm.bias",
        ] {
            let st = open("extra", safetensors_bytes(&w, &[(extra, &[4])]));
            let err = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone())
                .err()
                .unwrap_or_else(|| panic!("{extra}: loaded"));
            assert!(err.contains(extra), "{extra}: error does not name it: {err}");
        }
        // Tensors outside the prefix (the vision and audio towers) are not ours.
        let st = open("outside", safetensors_bytes(&w, &[("audio_tower.norm.bias", &[4])]));
        EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        // A missing tensor is refused too.
        let short = Weights(
            w.0.iter()
                .filter(|t| t.name != "layers.2.layer_scalar")
                .cloned()
                .collect(),
        );
        let st = open("missing", safetensors_bytes(&short, &[]));
        assert!(
            EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).is_err(),
            "missing layer_scalar: loaded"
        );
    });
}
