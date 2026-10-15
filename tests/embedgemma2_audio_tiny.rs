//! The whole EmbeddingGemma 2 audio tower and embedder
//! (`EmbedGemma2Audio::load` + `encode_audio`) on tiny random checkpoints,
//! against an f64 host forward written the way transformers writes it
//! (`tests/common/embedgemma2_audio.rs`). It runs everywhere `cargo test`
//! reaches a GPU, with no checkpoint: the real one is
//! `tests/embedgemma2_audio_model.rs`.
//!
//! Two configurations. `tiny` (hidden 64, head dim 16, chunk 4, window of 4
//! past frames) gives several blocks to a short clip; `wide` has the
//! checkpoint's attention geometry (chunk 12, context left 13, head dim 128)
//! at a small hidden size. Both have finite clip bounds, a gradient clip and a
//! logit cap that bite (checked: the reference with each relaxed disagrees with
//! the real one by far more than the bound), mel bins equal to the first conv's
//! channels, and a mask with every way to be padded: a clip all valid, one with
//! masked trailing frames, one with a hole in the middle (so invalid rows feed
//! later valid ones through the causal conv), a one-frame clip and a clip with
//! no valid frame.
//!
//! The checkpoint is built in memory as safetensors bytes (bf16 weights, scalar
//! bounds) and opened with `SafeTensors::from_bytes`.
//!
//! Bounds, fixed before any run (the same as the real-checkpoint test): the
//! subsample output, each layer's residual stream and the tower output within
//! `1e-4` of their largest magnitude; the soft tokens within `1e-4` max abs
//! (they are RMS-normalized then projected: unit scale); a clip alone and
//! inside the batch bit for bit equal.

mod common;

use common::embedgemma2_audio::{audio_forward, AudioRef, Weights};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2_audio::{
    sub_rows, AudioConfig, AudioInput, AudioOutput, EmbedGemma2Audio, MAX_FRAMES_PER_FORWARD,
};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16};

const LAYER_REL: f64 = 1e-4;
const SOFT_ABS: f64 = 1e-4;

const TINY: &str = r#"{"text_config":{"hidden_size":40},"audio_config":{
"attention_chunk_size":4,"attention_context_left":5,"attention_context_right":0,
"attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":4.0,"conv_kernel_size":5,
"gradient_clipping":2.5,"hidden_act":"silu","hidden_size":64,"model_type":"gemma4_audio",
"num_attention_heads":4,"num_hidden_layers":2,"output_proj_dims":48,"residual_weight":0.5,
"rms_norm_eps":1e-05,"subsampling_conv_channels":[32,8],"use_clipped_linears":true}}"#;

const WIDE: &str = r#"{"text_config":{"hidden_size":24},"audio_config":{
"attention_chunk_size":12,"attention_context_left":13,"attention_context_right":0,
"attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":4.0,"conv_kernel_size":5,
"gradient_clipping":3.0,"hidden_act":"silu","hidden_size":256,"model_type":"gemma4_audio",
"num_attention_heads":2,"num_hidden_layers":1,"output_proj_dims":48,"residual_weight":0.5,
"rms_norm_eps":1e-06,"subsampling_conv_channels":[32,32],"use_clipped_linears":true}}"#;

/// A tensor of the tiny checkpoint, kept already rounded to bf16.
struct Weight {
    name: String,
    shape: Vec<usize>,
    data: Vec<f64>,
}

fn round_bf16(v: f64) -> f64 {
    f64::from(bf16_bits_to_f32(f32_slice_to_bf16(&[v as f32])[0]))
}

/// Every tensor the tower and the embedder read, with the names and shapes of
/// the real checkpoint.
fn random_weights(cfg: &AudioConfig, seed: u64) -> Vec<Weight> {
    struct Gen {
        r: SplitMix,
        out: Vec<Weight>,
    }
    impl Gen {
        /// `offset + unit * scale` per element, rounded to bf16.
        fn push(&mut self, name: String, shape: Vec<usize>, scale: f64, offset: f64) {
            let n: usize = shape.iter().product();
            let data = (0..n)
                .map(|_| round_bf16(offset + f64::from(self.r.unit()) * scale))
                .collect();
            self.out.push(Weight { name, shape, data });
        }

        /// A clipped linear: its weight and four bounds that clip a visible
        /// share of the values (inputs and outputs are O(1)).
        fn clipped(&mut self, name: String, out_f: usize, in_f: usize) {
            self.push(
                format!("{name}.linear.weight"),
                vec![out_f, in_f],
                1.7 * (in_f as f64).powf(-0.5),
                0.0,
            );
            for (s, sign, lo, spread) in [
                ("input_min", -1.0, 1.3, 0.5),
                ("input_max", 1.0, 1.3, 0.5),
                ("output_min", -1.0, 1.6, 0.8),
                ("output_max", 1.0, 1.6, 0.8),
            ] {
                self.push(format!("{name}.{s}"), vec![], spread, lo);
                let last = self.out.last_mut().expect("just pushed");
                last.data.iter_mut().for_each(|v| *v *= sign);
            }
        }
    }
    let mut g = Gen {
        r: SplitMix::new(seed),
        out: Vec::new(),
    };
    let (h, hd, k) = (cfg.hidden as usize, cfg.head_dim() as usize, cfg.conv_kernel as usize);
    let (c0, c1) = (cfg.sub_channels[0] as usize, cfg.sub_channels[1] as usize);
    let lin = |fan_in: usize| 1.7 * (fan_in as f64).powf(-0.5);
    let sub = |s: &str| format!("audio_tower.subsample_conv_projection.{s}");
    g.push(sub("layer0.conv.weight"), vec![c0, 1, 3, 3], 0.5, 0.0);
    g.push(sub("layer0.norm.weight"), vec![c0], 0.3, 1.0);
    g.push(sub("layer1.conv.weight"), vec![c1, c0, 3, 3], lin(9 * c0), 0.0);
    g.push(sub("layer1.norm.weight"), vec![c1], 0.3, 1.0);
    g.push(
        sub("input_proj_linear.weight"),
        vec![h, cfg.proj_in() as usize],
        lin(cfg.proj_in() as usize),
        0.0,
    );
    for li in 0..cfg.layers as usize {
        let p = |s: &str| format!("audio_tower.layers.{li}.{s}");
        for ff in ["feed_forward1", "feed_forward2"] {
            g.clipped(p(&format!("{ff}.ffw_layer_1")), 4 * h, h);
            g.clipped(p(&format!("{ff}.ffw_layer_2")), h, 4 * h);
        }
        for n in ["q_proj", "k_proj", "v_proj", "post"] {
            g.clipped(p(&format!("self_attn.{n}")), h, h);
        }
        g.clipped(p("lconv1d.linear_start"), 2 * h, h);
        g.clipped(p("lconv1d.linear_end"), h, h);
        for ff in ["feed_forward1", "feed_forward2"] {
            g.push(p(&format!("{ff}.pre_layer_norm.weight")), vec![h], 0.3, 1.0);
            g.push(p(&format!("{ff}.post_layer_norm.weight")), vec![h], 0.3, 1.0);
        }
        for n in [
            "norm_pre_attn",
            "norm_post_attn",
            "norm_out",
            "lconv1d.pre_layer_norm",
            "lconv1d.conv_norm",
        ] {
            g.push(p(&format!("{n}.weight")), vec![h], 0.3, 1.0);
        }
        g.push(p("self_attn.relative_k_proj.weight"), vec![h, h], lin(h), 0.0);
        g.push(p("self_attn.per_dim_scale"), vec![hd], 0.6, 0.0);
        g.push(p("lconv1d.depthwise_conv1d.weight"), vec![h, 1, k], 0.6, 0.0);
    }
    g.push(
        "audio_tower.output_proj.weight".into(),
        vec![cfg.out_dims as usize, h],
        lin(h),
        0.0,
    );
    g.push(
        "audio_tower.output_proj.bias".into(),
        vec![cfg.out_dims as usize],
        0.2,
        0.0,
    );
    g.push(
        "embed_audio.embedding_projection.weight".into(),
        vec![cfg.embed_dim as usize, cfg.out_dims as usize],
        lin(cfg.out_dims as usize),
        0.0,
    );
    g.out
}

fn reference_weights(w: &[Weight]) -> Weights {
    w.iter().map(|t| (t.name.clone(), t.data.clone())).collect()
}

/// safetensors bytes of every weight (bf16), plus `extra` tensors (f32 zeros).
fn safetensors_bytes(w: &[Weight], extra: &[(&str, &[usize])]) -> Vec<u8> {
    let mut entries: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    for t in w {
        let bits = f32_slice_to_bf16(&t.data.iter().map(|&v| v as f32).collect::<Vec<_>>());
        let bytes = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        entries.push((t.name.clone(), "BF16", t.shape.clone(), bytes));
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

/// A clip of `frames` frames with `valid(frame)` as its mask.
fn clip(cfg: &AudioConfig, frames: usize, seed: u64, valid: impl Fn(usize) -> bool) -> AudioInput {
    let mut r = SplitMix::new(seed);
    AudioInput {
        input_features: (0..frames * cfg.mel_bins() as usize).map(|_| 2.0 * r.unit()).collect(),
        mask: (0..frames).map(valid).collect(),
    }
}

/// The clips of the main test; `scale` sets the lengths so the `wide`
/// configuration (blocks of 12) spans several blocks too.
fn clips(cfg: &AudioConfig, scale: usize) -> Vec<AudioInput> {
    vec![
        clip(cfg, 41 * scale, 1, |_| true),
        clip(cfg, 58 * scale, 2, |f| f < 45 * scale),
        clip(cfg, 30 * scale, 3, |f| !(8 * scale..16 * scale).contains(&f)),
        clip(cfg, 1, 4, |_| true),
        clip(cfg, 3, 5, |_| false),
    ]
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn max_abs(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - y).abs())
        .fold(0.0, f64::max)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every stage of clip `i`, bit for bit, in two outputs.
fn assert_bit_equal(what: &str, a: &AudioOutput, ia: usize, b: &AudioOutput, ib: usize) {
    assert_eq!(a.out_masks[ia], b.out_masks[ib], "{what}: out mask");
    assert_eq!(
        bits(&a.soft_tokens[ia]),
        bits(&b.soft_tokens[ib]),
        "{what}: soft tokens"
    );
    if !a.trace.is_empty() && !b.trace.is_empty() {
        let (ta, tb) = (&a.trace[ia], &b.trace[ib]);
        assert_eq!(bits(&ta.subsample), bits(&tb.subsample), "{what}: subsample");
        for (l, (x, y)) in ta.layers.iter().zip(&tb.layers).enumerate() {
            assert_eq!(bits(x), bits(y), "{what}: layer {l}");
        }
        assert_eq!(bits(&ta.tower_out), bits(&tb.tower_out), "{what}: tower out");
    }
}

fn check_against_reference(name: &str, cfg: &AudioConfig, w: &[Weight], scale: usize) {
    let st = open(name, safetensors_bytes(w, &[("vision_tower.unused.weight", &[3])]));
    let rw = reference_weights(w);
    let batch = clips(cfg, scale);
    let refs: Vec<AudioRef> = batch
        .iter()
        .map(|c| audio_forward(cfg, &rw, &to_f64(&c.input_features), &c.mask))
        .collect();
    // The mask patterns must really exercise padding.
    assert!(
        refs[1].out_mask.iter().any(|&m| !m) && refs[1].out_mask[0],
        "test bug: trailing mask"
    );
    let hole = &refs[2].out_mask;
    assert!(
        hole.iter()
            .position(|&m| !m)
            .is_some_and(|f| hole[f..].iter().any(|&m| m)),
        "test bug: the hole must be followed by valid rows"
    );
    assert!(
        refs[4].soft.is_empty() && !refs[4].out_mask[0],
        "test bug: no valid frame"
    );

    with_gpu(|rt| {
        let model = EmbedGemma2Audio::load(rt, &st, "", cfg.clone()).unwrap();
        let out = model.encode_audio(&batch, true).unwrap();
        assert_eq!(out.forwards, 1, "the tiny batch must share one forward");
        let (h, od, ed) = (cfg.hidden as usize, cfg.out_dims as usize, cfg.embed_dim as usize);
        for (i, (c, want)) in batch.iter().zip(&refs).enumerate() {
            let t = sub_rows(c.mask.len());
            assert_eq!(out.out_masks[i], want.out_mask, "{name} clip {i}: out mask");
            let tr = &out.trace[i];
            let mut stages: Vec<(String, &[f32], &[f64], usize)> = vec![
                ("subsample".into(), &tr.subsample, &want.subsample, h),
                ("tower_out".into(), &tr.tower_out, &want.tower, od),
            ];
            for (l, (got, w)) in tr.layers.iter().zip(&want.layers).enumerate() {
                stages.push((format!("layer {l}"), got, w, h));
            }
            for (stage, got, want_v, width) in stages {
                assert_eq!(got.len(), t * width, "{name} clip {i} {stage}: rows");
                let scale = want_v.iter().map(|v| v.abs()).fold(0.0, f64::max);
                let err = max_abs(got, want_v);
                eprintln!(
                    "{name} clip {i} ({} frames) {stage}: max abs {err:.3e} of scale {scale:.3e}",
                    c.mask.len()
                );
                assert!(
                    err <= LAYER_REL * scale,
                    "{name} clip {i} {stage}: max abs {err:.3e} > {LAYER_REL:.0e} * {scale:.3e}"
                );
            }
            assert_eq!(out.soft_tokens[i].len(), want.soft.len(), "{name} clip {i}: soft rows");
            assert_eq!(want.soft.len(), want.out_mask.iter().filter(|&&m| m).count() * ed);
            let err = max_abs(&out.soft_tokens[i], &want.soft);
            eprintln!("{name} clip {i}: soft max abs {err:.3e}");
            assert!(
                err <= SOFT_ABS,
                "{name} clip {i}: soft tokens max abs {err:.3e} > {SOFT_ABS:.0e}"
            );
        }

        // Each clip alone is bit for bit what it is in the batch.
        for (i, c) in batch.iter().enumerate() {
            let alone = model.encode_audio(std::slice::from_ref(c), true).unwrap();
            assert_bit_equal(&format!("{name} clip {i} alone vs batched"), &alone, 0, &out, i);
        }
        // Reordered, and a clip repeated.
        let order = [3usize, 1, 4, 1, 0, 2];
        let shuffled: Vec<AudioInput> = order.iter().map(|&i| batch[i].clone()).collect();
        let sh = model.encode_audio(&shuffled, true).unwrap();
        for (pos, &i) in order.iter().enumerate() {
            assert_bit_equal(&format!("{name} reordered {pos}"), &sh, pos, &out, i);
        }
        // Tracing changes nothing about the soft tokens.
        let plain = model.encode_audio(&batch, false).unwrap();
        assert!(plain.trace.is_empty());
        for i in 0..batch.len() {
            assert_eq!(
                bits(&plain.soft_tokens[i]),
                bits(&out.soft_tokens[i]),
                "{name} clip {i}: trace"
            );
        }

        // Every unzeroed activation poisoned with NaN: no bit moves.
        rt.set_poison_unzeroed(true);
        let poisoned = model.encode_audio(&batch, true);
        rt.set_poison_unzeroed(false);
        let poisoned = poisoned.unwrap();
        for i in 0..batch.len() {
            assert_bit_equal(&format!("{name} poisoned clip {i}"), &poisoned, i, &out, i);
        }
        assert!(out.soft_tokens.iter().flatten().all(|v| v.is_finite()));
    });
}

#[test]
fn tiny_audio_tower_matches_the_f64_forward() {
    let cfg = AudioConfig::from_config_json(TINY).unwrap();
    assert_eq!(
        (cfg.head_dim(), cfg.chunk, cfg.past(), cfg.mel_bins(), cfg.proj_in()),
        (16, 4, 4, 32, 64)
    );
    check_against_reference("tiny", &cfg, &random_weights(&cfg, 2026), 1);
}

#[test]
fn wide_audio_tower_has_the_checkpoints_attention_geometry() {
    let cfg = AudioConfig::from_config_json(WIDE).unwrap();
    assert_eq!(
        (cfg.head_dim(), cfg.chunk, cfg.past(), cfg.ctx(), cfg.rel_rows()),
        (128, 12, 12, 24, 13)
    );
    check_against_reference("wide", &cfg, &random_weights(&cfg, 7), 1);
}

/// The clamps, the gradient clip and the softcap are not decoration: with each
/// relaxed in the reference its output is far from the real one.
#[test]
fn the_bounds_bite() {
    let cfg = AudioConfig::from_config_json(TINY).unwrap();
    let w = random_weights(&cfg, 2026);
    let c = clip(&cfg, 41, 1, |_| true);
    let (feats, mask) = (to_f64(&c.input_features), &c.mask);
    let base = audio_forward(&cfg, &reference_weights(&w), &feats, mask);
    let scale = base.tower.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let far = |other: &AudioRef| -> f64 {
        base.tower
            .iter()
            .zip(&other.tower)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
            / scale
    };

    let mut no_bounds = reference_weights(&w);
    for (k, v) in no_bounds.iter_mut() {
        if k.ends_with("_min") {
            v[0] = -1e6;
        } else if k.ends_with("_max") {
            v[0] = 1e6;
        }
    }
    let d_bounds = far(&audio_forward(&cfg, &no_bounds, &feats, mask));
    let mut loose = cfg.clone();
    loose.gradient_clip = 1e6;
    let d_clip = far(&audio_forward(&loose, &reference_weights(&w), &feats, mask));
    let mut uncapped = cfg.clone();
    uncapped.logit_cap = 1e6;
    let d_cap = far(&audio_forward(&uncapped, &reference_weights(&w), &feats, mask));
    eprintln!("relative change of the tower output without: bounds {d_bounds:.2e}, gradient clip {d_clip:.2e}, softcap {d_cap:.2e}");
    for (what, d) in [("clip bounds", d_bounds), ("gradient clip", d_clip), ("softcap", d_cap)] {
        assert!(
            d > 100.0 * LAYER_REL,
            "{what} change the output by only {d:.2e}: they do not bite"
        );
    }
}

/// More frames than one forward holds: the clips split across forwards, which
/// queue back to back and are waited for once, and each clip is bit for bit
/// what it is alone.
#[test]
fn a_batch_past_the_frame_budget_splits_into_forwards() {
    let cfg = AudioConfig::from_config_json(TINY).unwrap();
    let w = random_weights(&cfg, 11);
    let st = open("split", safetensors_bytes(&w, &[]));
    let half = MAX_FRAMES_PER_FORWARD / 2 + 7;
    let batch = vec![
        clip(&cfg, half, 21, |f| f % 11 != 0),
        clip(&cfg, half, 22, |_| true),
        clip(&cfg, 90, 23, |f| f < 70),
    ];
    with_gpu(|rt| {
        let model = EmbedGemma2Audio::load(rt, &st, "", cfg.clone()).unwrap();
        let all = model.encode_audio(&batch, false).unwrap();
        assert_eq!(all.forwards, 2, "{} frames cannot share a forward", 2 * half + 90);
        for (i, c) in batch.iter().enumerate() {
            let alone = model.encode_audio(std::slice::from_ref(c), false).unwrap();
            assert_bit_equal(&format!("split clip {i}"), &alone, 0, &all, i);
            assert_eq!(
                all.soft_tokens[i].len(),
                all.out_masks[i].iter().filter(|&&m| m).count() * cfg.embed_dim as usize
            );
        }
        rt.set_poison_unzeroed(true);
        let poisoned = model.encode_audio(&batch, false);
        rt.set_poison_unzeroed(false);
        let poisoned = poisoned.unwrap();
        for i in 0..batch.len() {
            assert_bit_equal(&format!("split poisoned {i}"), &poisoned, i, &all, i);
        }
    });
}

#[test]
fn encode_audio_refuses_what_it_cannot_run() {
    let cfg = AudioConfig::from_config_json(TINY).unwrap();
    let w = random_weights(&cfg, 3);
    let st = open("refuse", safetensors_bytes(&w, &[]));
    let good = clip(&cfg, 12, 1, |_| true);
    with_gpu(|rt| {
        let model = EmbedGemma2Audio::load(rt, &st, "", cfg.clone()).unwrap();
        let err = |batch: &[AudioInput]| model.encode_audio(batch, false).expect_err("accepted");
        assert!(err(&[]).contains("empty batch"));
        let mut bad = good.clone();
        bad.mask.clear();
        bad.input_features.clear();
        assert!(err(&[bad]).contains("no frames"));
        let mut bad = good.clone();
        bad.input_features.pop();
        assert!(err(&[good.clone(), bad]).contains("clip 1"));
        let mut bad = good.clone();
        bad.input_features[5] = f32::NAN;
        assert!(err(&[bad]).contains("not finite"));
        let mut bad = good.clone();
        bad.input_features[0] = f32::INFINITY;
        assert!(err(&[bad]).contains("not finite"));
        let too_long = clip(&cfg, MAX_FRAMES_PER_FORWARD + 1, 2, |_| true);
        assert!(err(&[too_long]).contains("split it"));
        let many = vec![good.clone(); tessl::embedgemma2_audio::MAX_CLIPS + 1];
        assert!(err(&many).contains("batch limit"));
        // A different mel width than the first conv's channels.
        let mut bad = good.clone();
        bad.input_features.extend([0.0; 7]);
        assert!(err(&[bad]).contains("mel bins"));
        // The model is still usable after refusals.
        assert!(model.encode_audio(std::slice::from_ref(&good), false).is_ok());
        // Relaxed f32 GEMMs would not be exact: refused by name.
        rt.set_relaxed_precision(true);
        let relaxed = model.encode_audio(std::slice::from_ref(&good), false);
        rt.set_relaxed_precision(false);
        if rt.has_tensorops() {
            assert!(relaxed.expect_err("relaxed accepted").contains("exact-f32"));
        }
    });
}

#[test]
fn load_refuses_a_checkpoint_that_does_not_match() {
    let cfg = AudioConfig::from_config_json(TINY).unwrap();
    let w = random_weights(&cfg, 3);
    with_gpu(|rt| {
        let load = |tag: &str, weights: &[Weight], extra: &[(&str, &[usize])]| {
            let st = open(tag, safetensors_bytes(weights, extra));
            EmbedGemma2Audio::load(rt, &st, "", cfg.clone()).err()
        };
        assert!(load("ok", &w, &[("vision_tower.x", &[2]), ("language_model.y", &[2])]).is_none());
        // A tensor the forward does not read, under either prefix.
        let e = load("extra tower", &w, &[("audio_tower.layers.0.self_attn.bias", &[4])]).expect("accepted");
        assert!(e.contains("audio_tower.layers.0.self_attn.bias"), "{e}");
        let e = load(
            "extra embed",
            &w,
            &[("embed_audio.embedding_pre_projection_norm.weight", &[4])],
        )
        .expect("accepted");
        assert!(e.contains("embed_audio."), "{e}");
        // A missing tensor, and a wrong shape.
        let missing: Vec<Weight> = w
            .iter()
            .filter(|t| t.name != "audio_tower.layers.1.lconv1d.conv_norm.weight")
            .map(|t| Weight {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data: t.data.clone(),
            })
            .collect();
        let e = load("missing", &missing, &[]).expect("accepted");
        assert!(e.contains("lconv1d.conv_norm"), "{e}");
        let mut reshaped: Vec<Weight> = w
            .iter()
            .map(|t| Weight {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data: t.data.clone(),
            })
            .collect();
        let t = reshaped
            .iter_mut()
            .find(|t| t.name == "audio_tower.output_proj.bias")
            .unwrap();
        t.shape = vec![t.data.len() / 2, 2];
        let e = load("reshaped", &reshaped, &[]).expect("accepted");
        assert!(e.contains("output_proj.bias"), "{e}");
        // A prefix other than the tensors' is "missing".
        let st = open("prefix", safetensors_bytes(&w, &[]));
        assert!(EmbedGemma2Audio::load(rt, &st, "model.", cfg.clone()).is_err());
    });
}
