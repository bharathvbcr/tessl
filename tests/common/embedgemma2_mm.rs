//! The tiny multimodal checkpoint: the text encoder of `common::embedgemma2`
//! and a tiny vision and audio tower in one `config.json` and one safetensors
//! file, as `google/embeddinggemma-2` ships them, for the tests of
//! `tessl::embedgemma2_mm::EmbedGemma2Multimodal`, its C ABI entries and the
//! processor-shaped inputs they take.
//!
//! The tower tensors are random with the names, shapes and scales every tiny
//! tower test uses: this is their one generator ([`vision_weights`] and
//! [`audio_weights`], which keep the stored values as f64 for the f64
//! references of `tests/embedgemma2_vision_tiny.rs` and
//! `tests/embedgemma2_audio_tiny.rs`; the facade's tests take them as
//! safetensors entries, [`vision_tensors`] and [`audio_tensors`]). There is no
//! f64 reference for the facade itself because it is a composition, and the
//! oracle of its tests is the composition done by hand. The facade's checkpoint
//! stores every tower tensor as bf16, as the real one does
//! ([`VisionStorage::CHECKPOINT`]); the vision tests also store projections
//! and norms as f32 (see [`VisionStorage`]).

use super::embedgemma2::{pack_safetensors, text_tensors, weights_for, RawTensor, CONFIG};
use super::SplitMix;
use tessl::embedgemma2::EmbedGemma2Config;
use tessl::embedgemma2_audio::{AudioConfig, AudioInput};
use tessl::embedgemma2_vision::VisionConfig;
use tessl::gemm::GemmOperands;
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16};

/// Placeholder ids: past the tiny text vocabulary of 48, as `embedgemma2_mm_tiny`
/// puts them.
pub const IMAGE: u32 = 60;
pub const AUDIO: u32 = 61;
pub const VIDEO: u32 = 62;

/// Two layers of two heads of 64 (the head size the bf16 attention kernel is compiled for, so one checkpoint serves both modes), hidden 96, patch 4 (patch width 48), a 2x2
/// pooling cell, a 128-row position table.
pub const VISION_JSON: &str = r#"{"model_type":"gemma4_vision","hidden_activation":"gelu_pytorch_tanh","hidden_size":96,
"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64,"intermediate_size":160,
"patch_size":4,"pooling_kernel_size":2,"position_embedding_size":128,"rms_norm_eps":1e-06,
"attention_bias":false,"attention_dropout":0.0,"standardize":false,"use_clipped_linears":false,
"rope_parameters":{"rope_theta":100.0,"rope_type":"axial"}}"#;

/// Two conformer layers of four heads, hidden 64, blocks of 4 queries.
pub const AUDIO_JSON: &str = r#"{
"attention_chunk_size":4,"attention_context_left":5,"attention_context_right":0,
"attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":4.0,"conv_kernel_size":5,
"gradient_clipping":2.5,"hidden_act":"silu","hidden_size":64,"model_type":"gemma4_audio",
"num_attention_heads":4,"num_hidden_layers":2,"output_proj_dims":48,"residual_weight":0.5,
"rms_norm_eps":1e-05,"subsampling_conv_channels":[32,8],"use_clipped_linears":true}"#;

/// The ids `config.json` carries at the top level and in `text_config`.
const MM_IDS: &str = r#""image_token_id":60,"audio_token_id":61,"video_token_id":62,"boi_token_id":40,"eoi_token_id":41,"boa_token_id":42,"eoa_token_index":43,"#;

/// The tiny `config.json`: the text encoder's, the multimodal ids, and both
/// towers (whose soft tokens are as wide as the text encoder's hidden size, 64).
pub fn config_json() -> String {
    let top = format!("{{{MM_IDS}\"vision_config\":{VISION_JSON},\"audio_config\":{AUDIO_JSON},");
    // The text ids first: the audio config also has a `"hidden_size":64,`.
    CONFIG
        .replacen(
            r#""hidden_size":64,"#,
            r#""pad_token_id":0,"bos_token_id":2,"eos_token_id":1,"hidden_size":64,"#,
            1,
        )
        .replacen('{', &top, 1)
}

/// The tiny `config.json` with the towers' configs only (no multimodal ids):
/// what a text-only config plus towers looks like.
pub fn config_json_without_ids() -> String {
    let top = format!("{{\"vision_config\":{VISION_JSON},\"audio_config\":{AUDIO_JSON},");
    CONFIG.replacen('{', &top, 1)
}

pub struct Configs {
    pub text: EmbedGemma2Config,
    pub vision: VisionConfig,
    pub audio: AudioConfig,
}

pub fn configs() -> Configs {
    let json = config_json();
    Configs {
        text: EmbedGemma2Config::from_config_json(&json).unwrap(),
        vision: VisionConfig::from_config_json(&json).unwrap(),
        audio: AudioConfig::from_config_json(&json).unwrap(),
    }
}

/// A tensor of a tiny tower checkpoint: its name, shape and values as stored.
/// A bf16 tensor's values are kept already rounded to bf16, so a host
/// reference computed from `data` reads exactly what the device does; an f32
/// tensor's are the f64 draws (stored as f32, as the device reads them).
#[derive(Clone, Debug)]
pub struct TowerWeight {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub bf16: bool,
}

impl TowerWeight {
    /// The safetensors entry: BF16 or F32 bytes of `data`.
    pub fn raw(&self) -> RawTensor {
        let (dtype, bytes) = if self.bf16 {
            let bits = f32_slice_to_bf16(&self.data.iter().map(|&v| v as f32).collect::<Vec<_>>());
            ("BF16", bits.iter().flat_map(|b| b.to_le_bytes()).collect())
        } else {
            (
                "F32",
                self.data.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect(),
            )
        };
        RawTensor {
            name: self.name.clone(),
            dtype,
            shape: self.shape.clone(),
            bytes,
        }
    }
}

/// The safetensors entries of `weights`, in order.
pub fn raw_tensors(weights: &[TowerWeight]) -> Vec<RawTensor> {
    weights.iter().map(TowerWeight::raw).collect()
}

/// `v` rounded to bf16 (through f32, as the checkpoint's writer does).
pub fn round_bf16(v: f64) -> f64 {
    f64::from(bf16_bits_to_f32(f32_slice_to_bf16(&[v as f32])[0]))
}

/// A random tensor stream: `offset + unit * scale` per element, in the order
/// the towers' tests have always drawn it (a given seed makes the same values
/// whatever the storage).
struct Gen {
    r: SplitMix,
    out: Vec<TowerWeight>,
}

impl Gen {
    /// A tensor stored as `bf16` (rounded to it) or f32.
    fn push_as(&mut self, name: String, shape: Vec<usize>, scale: f64, offset: f64, bf16: bool) {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|_| {
                let v = offset + f64::from(self.r.unit()) * scale;
                if bf16 {
                    round_bf16(v)
                } else {
                    v
                }
            })
            .collect();
        self.out.push(TowerWeight {
            name,
            shape,
            data,
            bf16,
        });
    }

    /// A bf16 tensor.
    fn push(&mut self, name: String, shape: Vec<usize>, scale: f64, offset: f64) {
        self.push_as(name, shape, scale, offset, true);
    }

    /// A clipped linear: its weight and four bounds that clip a visible share
    /// of the values (inputs and outputs are O(1)).
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

/// How the vision tower's tensors are stored. The patch projection and the
/// position table are always bf16, as the real checkpoint stores them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionStorage {
    /// The attention and MLP projections and the embedder's: bf16 (what
    /// `GemmOperands::Bf16` needs and the checkpoint has) or f32.
    pub projections_bf16: bool,
    /// The norm weights (layer norms, `q_norm`, `k_norm`): bf16 or f32.
    pub norms_bf16: bool,
}

impl VisionStorage {
    /// Everything bf16, as `google/embeddinggemma-2` ships it: what the
    /// facade's tiny checkpoint holds.
    pub const CHECKPOINT: Self = Self {
        projections_bf16: true,
        norms_bf16: true,
    };
}

/// Every tensor the vision tower and its embedder read, with the values the
/// tower sees (see [`TowerWeight`]).
pub fn vision_weights(cfg: &VisionConfig, seed: u64, storage: VisionStorage) -> Vec<TowerWeight> {
    let (proj, norm) = (storage.projections_bf16, storage.norms_bf16);
    let mut g = Gen {
        r: SplitMix::new(seed),
        out: Vec::new(),
    };
    let (h, inter, pd, d) = (
        cfg.hidden as usize,
        cfg.intermediate as usize,
        cfg.patch_dim() as usize,
        cfg.head_dim as usize,
    );
    let qw = (cfg.heads * cfg.head_dim) as usize;
    let lin = |fan_in: usize| (fan_in as f64).powf(-0.5);
    // Patch pixels are in [-1, 1] after scaling; a unit-scale projection keeps h O(1).
    g.push(
        "vision_tower.patch_embedder.input_proj.weight".into(),
        vec![h, pd],
        lin(pd),
        0.0,
    );
    g.push(
        "vision_tower.patch_embedder.position_embedding_table".into(),
        vec![2, cfg.position_embedding_size as usize, h],
        0.5,
        0.0,
    );
    for i in 0..cfg.layers {
        let p = |n: &str| format!("vision_tower.encoder.layers.{i}.{n}");
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            g.push_as(p(&format!("{n}.weight")), vec![h], 0.3, 1.0, norm);
        }
        for n in ["q_proj", "k_proj", "v_proj"] {
            g.push_as(
                p(&format!("self_attn.{n}.linear.weight")),
                vec![qw, h],
                lin(h),
                0.0,
                proj,
            );
        }
        g.push_as(p("self_attn.o_proj.linear.weight"), vec![h, qw], lin(qw), 0.0, proj);
        // Scale 1.0 attention over RMS-normed d-wide heads: weights near
        // d^-0.25 keep q.k O(1), so the softmax spreads over the keys.
        let qk = (d as f64).powf(-0.25);
        g.push_as(p("self_attn.q_norm.weight"), vec![d], 0.3 * qk, qk, norm);
        g.push_as(p("self_attn.k_norm.weight"), vec![d], 0.3 * qk, qk, norm);
        g.push_as(p("mlp.gate_proj.linear.weight"), vec![inter, h], lin(h), 0.0, proj);
        g.push_as(p("mlp.up_proj.linear.weight"), vec![inter, h], lin(h), 0.0, proj);
        g.push_as(p("mlp.down_proj.linear.weight"), vec![h, inter], lin(inter), 0.0, proj);
    }
    g.push_as(
        "embed_vision.embedding_projection.weight".into(),
        vec![cfg.text_hidden as usize, h],
        lin(h),
        0.0,
        proj,
    );
    g.out
}

/// [`vision_weights`] as safetensors entries, everything bf16 as the
/// checkpoint stores it ([`VisionStorage::CHECKPOINT`]).
pub fn vision_tensors(cfg: &VisionConfig, seed: u64) -> Vec<RawTensor> {
    raw_tensors(&vision_weights(cfg, seed, VisionStorage::CHECKPOINT))
}

/// [`audio_weights`] as safetensors entries.
pub fn audio_tensors(cfg: &AudioConfig, seed: u64) -> Vec<RawTensor> {
    raw_tensors(&audio_weights(cfg, seed))
}

/// Every tensor the audio tower and its embedder read, all bf16 as the
/// checkpoint's are, with the values the tower sees (see [`TowerWeight`]).
pub fn audio_weights(cfg: &AudioConfig, seed: u64) -> Vec<TowerWeight> {
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

/// The whole tiny checkpoint's safetensors bytes: text (stored for `operands`),
/// vision and audio.
pub fn checkpoint_bytes(c: &Configs, seed: u64, operands: GemmOperands) -> Vec<u8> {
    let mut tensors = text_tensors(&weights_for(&c.text, seed, operands));
    tensors.extend(vision_tensors(&c.vision, seed + 1));
    tensors.extend(audio_tensors(&c.audio, seed + 2));
    pack_safetensors(&tensors)
}

/// One synthetic image as the processor lays it out: a `w x h` raster of
/// patches then `pad` padding patches `(-1, -1)`, `pd` pixel values in `[0, 1]`
/// per patch.
pub struct Image {
    pub pixel_values: Vec<f32>,
    pub position_ids: Vec<i32>,
}

impl Image {
    pub fn new(w: i32, h: i32, pad: usize, pd: usize, seed: u64) -> Self {
        let mut r = SplitMix::new(seed);
        let n = (w * h) as usize;
        let mut position_ids = Vec::with_capacity(2 * (n + pad));
        for y in 0..h {
            for x in 0..w {
                position_ids.extend([x, y]);
            }
        }
        position_ids.extend(std::iter::repeat_n(-1, 2 * pad));
        let pixel_values = (0..(n + pad) * pd).map(|_| (r.unit() + 1.0) * 0.5).collect();
        Self {
            pixel_values,
            position_ids,
        }
    }

    pub fn input(&self) -> tessl::embedgemma2_vision::ImageInput<'_> {
        tessl::embedgemma2_vision::ImageInput {
            pixel_values: &self.pixel_values,
            position_ids: &self.position_ids,
        }
    }
}

/// A clip of `frames` frames with `valid(frame)` as its mask.
pub fn clip(cfg: &AudioConfig, frames: usize, seed: u64, valid: impl Fn(usize) -> bool) -> AudioInput {
    let mut r = SplitMix::new(seed);
    AudioInput {
        input_features: (0..frames * cfg.mel_bins() as usize).map(|_| 2.0 * r.unit()).collect(),
        mask: (0..frames).map(valid).collect(),
    }
}
