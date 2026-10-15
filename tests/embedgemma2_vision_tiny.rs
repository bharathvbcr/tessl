//! EmbeddingGemma 2's vision tower and multimodal embedder
//! (`EmbedGemma2VisionModel::load` + `encode_images`) on a tiny random
//! checkpoint, against an f64 host forward (`tests/common/embedgemma2_vision.rs`,
//! composed from the text encoder's references). It runs wherever `cargo test`
//! reaches a GPU, with no checkpoint: the real one is
//! `tests/embedgemma2_vision_model.rs`.
//!
//! The config is two layers of three heads of 32 (hidden 96, patch 4 so the
//! patch width 48 differs from the hidden width, a 2x2 pooling cell, a 128-row
//! position table), and four images that exercise what the real ones do at
//! 2520 patches: a 8x6 grid with 16 padding patches, a 5x4 grid whose odd
//! width makes pooling cells collide (transformers' `max_x // k`), a single
//! patch, and a full 2x2 grid with no padding. They run alone and as one
//! ragged batch.
//!
//! Bounds, fixed before any run (the same as the real-checkpoint test): each
//! layer's residual stream and the pooled output within `1e-4` of their
//! largest magnitude; soft tokens within `1e-4` max abs; each image alone vs
//! inside the batch bit for bit (the padded keys are masked and the padded
//! rows are zero, so nothing but a different GEMM tiling could move them).

mod common;

use common::embedgemma2_vision::{vision_forward, VisionDims, VisionRef};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, VisionConfig, MAX_BATCH_PATCHES};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16};

const LAYER_REL: f64 = 1e-4;
const SOFT_ABS: f64 = 1e-4;

const CONFIG: &str = r#"{"model_type":"embedding_gemma2",
"text_config":{"hidden_size":48,"model_type":"embedding_gemma2_text"},
"vision_config":{"model_type":"gemma4_vision","hidden_activation":"gelu_pytorch_tanh","hidden_size":96,
"num_hidden_layers":2,"num_attention_heads":3,"num_key_value_heads":3,"head_dim":32,"intermediate_size":160,
"patch_size":4,"pooling_kernel_size":2,"position_embedding_size":128,"rms_norm_eps":1e-06,
"attention_bias":false,"attention_dropout":0.0,"standardize":false,"use_clipped_linears":false,
"rope_parameters":{"rope_theta":100.0,"rope_type":"axial"}}}"#;

/// A tensor of the tiny checkpoint: full name, shape, values. `bf16` tensors
/// are stored as bf16 and kept here already rounded to it.
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

fn dims(cfg: &VisionConfig) -> VisionDims {
    VisionDims {
        hidden: cfg.hidden as usize,
        layers: cfg.layers as usize,
        heads: cfg.heads as usize,
        head_dim: cfg.head_dim as usize,
        inter: cfg.intermediate as usize,
        patch_dim: cfg.patch_dim() as usize,
        pool_k: cfg.pooling_kernel as usize,
        table_len: cfg.position_embedding_size as usize,
        theta: f64::from(cfg.rope_theta),
        text_hidden: cfg.text_hidden as usize,
    }
}

fn random_weights(cfg: &VisionConfig, seed: u64) -> Weights {
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
    let (h, inter, pd, d) = (
        cfg.hidden as usize,
        cfg.intermediate as usize,
        cfg.patch_dim() as usize,
        cfg.head_dim as usize,
    );
    let qw = (cfg.heads * cfg.head_dim) as usize;
    let lin_scale = |fan_in: usize| (fan_in as f64).powf(-0.5);
    // Patch pixels are in [-1, 1] after scaling; unit-scale projection keeps h O(1).
    push(
        "vision_tower.patch_embedder.input_proj.weight".into(),
        vec![h, pd],
        lin_scale(pd),
        0.0,
        true,
    );
    push(
        "vision_tower.patch_embedder.position_embedding_table".into(),
        vec![2, cfg.position_embedding_size as usize, h],
        0.5,
        0.0,
        true,
    );
    for i in 0..cfg.layers {
        let p = |n: &str| format!("vision_tower.encoder.layers.{i}.{n}");
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            push(p(&format!("{n}.weight")), vec![h], 0.3, 1.0, false);
        }
        for n in ["q_proj", "k_proj", "v_proj"] {
            push(
                p(&format!("self_attn.{n}.linear.weight")),
                vec![qw, h],
                lin_scale(h),
                0.0,
                false,
            );
        }
        push(
            p("self_attn.o_proj.linear.weight"),
            vec![h, qw],
            lin_scale(qw),
            0.0,
            false,
        );
        // Scale 1.0 attention over RMS-normed d-wide heads: weights near
        // d^-0.25 keep q.k O(1), so the softmax spreads over the keys.
        let qk = (d as f64).powf(-0.25);
        push(p("self_attn.q_norm.weight"), vec![d], 0.3 * qk, qk, false);
        push(p("self_attn.k_norm.weight"), vec![d], 0.3 * qk, qk, false);
        push(
            p("mlp.gate_proj.linear.weight"),
            vec![inter, h],
            lin_scale(h),
            0.0,
            false,
        );
        push(p("mlp.up_proj.linear.weight"), vec![inter, h], lin_scale(h), 0.0, false);
        push(
            p("mlp.down_proj.linear.weight"),
            vec![h, inter],
            lin_scale(inter),
            0.0,
            false,
        );
    }
    push(
        "embed_vision.embedding_projection.weight".into(),
        vec![cfg.text_hidden as usize, h],
        lin_scale(h),
        0.0,
        false,
    );
    Weights(out)
}

/// safetensors bytes: every weight, plus `extra` tensors named as given (f32
/// zeros of the given shape), minus the ones named in `skip`.
fn safetensors_bytes(w: &Weights, extra: &[(&str, &[usize])], skip: &[&str]) -> Vec<u8> {
    let mut entries: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    for t in w.0.iter().filter(|t| !skip.contains(&t.name.as_str())) {
        let bytes: Vec<u8> = if t.bf16 {
            f32_slice_to_bf16(&t.data.iter().map(|&v| v as f32).collect::<Vec<_>>())
                .iter()
                .flat_map(|b| b.to_le_bytes())
                .collect()
        } else {
            t.data.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect()
        };
        entries.push((
            t.name.clone(),
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

/// One synthetic image: a `w x h` raster of patches then `pad` padding
/// patches, `pd` pixel values per patch in `[0, 1]`.
struct Image {
    pixels: Vec<f32>,
    positions: Vec<i32>,
}

impl Image {
    fn new(w: i32, h: i32, pad: usize, pd: usize, seed: u64) -> Self {
        let mut r = SplitMix::new(seed);
        let n = (w * h) as usize;
        let mut positions = Vec::with_capacity(2 * (n + pad));
        for y in 0..h {
            for x in 0..w {
                positions.extend([x, y]);
            }
        }
        positions.extend(std::iter::repeat_n(-1, 2 * pad));
        // Padding patches carry arbitrary pixels, as the processor's do not
        // matter: the tower never reads them.
        let pixels = (0..(n + pad) * pd).map(|_| (r.unit() + 1.0) * 0.5).collect();
        Self { pixels, positions }
    }

    fn input(&self) -> ImageInput<'_> {
        ImageInput {
            pixel_values: &self.pixels,
            position_ids: &self.positions,
        }
    }
}

fn images(pd: usize) -> Vec<Image> {
    vec![
        Image::new(8, 6, 16, pd, 1), // 48 real, P = 64, 12 cells
        Image::new(5, 4, 28, pd, 2), // 20 real, P = 48, odd width: colliding cells
        Image::new(1, 1, 3, pd, 3),  // 1 real, P = 4
        Image::new(2, 2, 0, pd, 4),  // 4 real, no padding
    ]
}

fn max_abs(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - y).abs())
        .fold(0.0, f64::max)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `got` within `rel` of `want`'s largest magnitude; the observed ratio is printed.
fn check_rel(what: &str, got: &[f32], want: &[f64], rel: f64) {
    let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let err = max_abs(got, want);
    eprintln!(
        "{what}: max abs {err:.3e} of scale {scale:.3e} ({:.2e} relative)",
        err / scale
    );
    assert!(
        err <= rel * scale,
        "{what}: max abs {err:.3e} > {rel:.0e} * {scale:.3e}"
    );
}

fn check_against_reference(tag: &str, out: &tessl::embedgemma2_vision::VisionOutput, i: usize, want: &VisionRef) {
    assert_eq!(out.patches[i], want.n, "{tag}: real patches");
    assert_eq!(out.tokens[i], want.tokens, "{tag}: soft tokens");
    let t = &out.trace;
    check_rel(
        &format!("{tag} patch_embed"),
        &t.patch_embed[i],
        &want.patch_embed,
        LAYER_REL,
    );
    for (l, w) in want.layers.iter().enumerate() {
        check_rel(&format!("{tag} layer {l}"), &t.layers[l][i], w, LAYER_REL);
    }
    check_rel(&format!("{tag} pooled"), &t.pooled[i], &want.pooled, LAYER_REL);
    let err = max_abs(&out.soft[i], &want.soft);
    eprintln!("{tag} soft: max abs {err:.3e}");
    assert!(err <= SOFT_ABS, "{tag} soft: max abs {err:.3e} > {SOFT_ABS:.0e}");
}

#[test]
fn tiny_vision_matches_the_f64_forward_alone_and_batched() {
    let cfg = VisionConfig::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 2026);
    let st = open(
        "ok",
        safetensors_bytes(&w, &[("language_model.unused.weight", &[3])], &[]),
    );
    let d = dims(&cfg);
    let imgs = images(d.patch_dim);
    let refs: Vec<VisionRef> = imgs
        .iter()
        .map(|im| vision_forward(&d, &|n| w.get(n), &im.pixels, &im.positions))
        .collect();
    assert_eq!(
        refs.iter().map(|r| r.tokens).collect::<Vec<_>>(),
        [12, 5, 1, 1],
        "test bug: the images' soft token counts changed"
    );

    with_gpu(|rt| {
        let model = EmbedGemma2VisionModel::load(rt, &st, cfg.clone()).unwrap();

        // 1. Each image alone, layer by layer.
        let mut alone = Vec::new();
        for (i, im) in imgs.iter().enumerate() {
            let out = model.encode_images(&[im.input()], true).unwrap();
            assert_eq!(out.forwards, 1);
            assert_eq!(out.trace.layers.len(), cfg.layers as usize);
            check_against_reference(&format!("image {i} alone"), &out, 0, &refs[i]);
            assert_eq!(out.soft[0].len(), out.tokens[0] * cfg.text_hidden as usize);
            alone.push(out.soft.into_iter().next().unwrap());
        }

        // 2. All four in one forward (ragged: 48, 20, 1, 4 real patches),
        //    against the f64 forward and against themselves alone, bit for bit.
        let all: Vec<ImageInput<'_>> = imgs.iter().map(Image::input).collect();
        let batched = model.encode_images(&all, true).unwrap();
        assert_eq!(batched.forwards, 1, "the tiny batch must share one forward");
        for i in 0..imgs.len() {
            check_against_reference(&format!("image {i} batched"), &batched, i, &refs[i]);
            assert_eq!(
                bits(&batched.soft[i]),
                bits(&alone[i]),
                "image {i}: batched soft tokens differ from alone"
            );
        }

        // 3. The input order is irrelevant: row i of a reversed call is its image's.
        let reversed: Vec<ImageInput<'_>> = imgs.iter().rev().map(Image::input).collect();
        let rev = model.encode_images(&reversed, false).unwrap();
        assert!(rev.trace.layers.iter().all(Vec::is_empty) && rev.trace.pooled.is_empty());
        for (i, a) in alone.iter().enumerate() {
            assert_eq!(bits(&rev.soft[imgs.len() - 1 - i]), bits(a), "image {i}: reversed call");
        }
    });
}

/// Every unzeroed activation poisoned with NaN changes no bit of any soft
/// token: no kernel reads an element its forward did not write (the padded
/// rows, the K/V capacity past `seq`, the pooled rows).
#[test]
fn poisoned_unzeroed_activations_change_nothing() {
    let cfg = VisionConfig::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 7);
    let st = open("poison", safetensors_bytes(&w, &[], &[]));
    let imgs = images(cfg.patch_dim() as usize);
    let all: Vec<ImageInput<'_>> = imgs.iter().map(Image::input).collect();
    with_gpu(|rt| {
        let model = EmbedGemma2VisionModel::load(rt, &st, cfg.clone()).unwrap();
        let clean = model.encode_images(&all, true).unwrap();
        rt.set_poison_unzeroed(true);
        let poisoned = model.encode_images(&all, true);
        rt.set_poison_unzeroed(false);
        let poisoned = poisoned.unwrap();
        for i in 0..imgs.len() {
            assert_eq!(bits(&poisoned.soft[i]), bits(&clean.soft[i]), "image {i} soft");
            assert!(
                clean.soft[i].iter().all(|v| v.is_finite()),
                "image {i}: non-finite soft token"
            );
            for l in 0..cfg.layers as usize {
                assert_eq!(
                    bits(&poisoned.trace.layers[l][i]),
                    bits(&clean.trace.layers[l][i]),
                    "image {i} layer {l}"
                );
            }
        }
    });
}

/// A call past `MAX_BATCH_PATCHES` padded patches splits into several
/// forwards queued behind one wait; each image still equals itself alone, bit
/// for bit, with the activations poisoned and the runtime asynchronous.
#[test]
fn a_call_that_splits_into_forwards_matches_each_image_alone() {
    let cfg = VisionConfig::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 11);
    let st = open("split", safetensors_bytes(&w, &[], &[]));
    let pd = cfg.patch_dim() as usize;
    // Three images of 6000, 9000 and 5000 real patches (125 columns wide on
    // the 128-row table, no padding), in an order that is not their sorted
    // one: the 9000 runs alone (two rows of it would be 18000 padded patches,
    // over the 16384 budget), the other two share a forward of 2 x 6000.
    let mid = Image::new(125, 48, 0, pd, 21);
    let big = Image::new(125, 72, 0, pd, 22);
    let small = Image::new(125, 40, 0, pd, 23);
    let all = [mid.input(), big.input(), small.input()];
    const _: () = assert!(2 * 9000 > MAX_BATCH_PATCHES && 2 * 6000 <= MAX_BATCH_PATCHES);
    with_gpu(|rt| {
        let model = EmbedGemma2VisionModel::load(rt, &st, cfg.clone()).unwrap();
        let sync = model.encode_images(&all, false).unwrap();
        assert_eq!(sync.forwards, 2, "test bug: the call must split in two");

        rt.set_async_encode(true).unwrap();
        rt.set_poison_unzeroed(true);
        let queued = model.encode_images(&all, false);
        rt.set_poison_unzeroed(false);
        rt.set_async_encode(false).unwrap();
        let queued = queued.unwrap();
        for (i, img) in all.iter().enumerate() {
            assert_eq!(bits(&queued.soft[i]), bits(&sync.soft[i]), "image {i}: async vs sync");
            let alone = model.encode_images(&[*img], false).unwrap();
            assert_eq!(
                bits(&sync.soft[i]),
                bits(&alone.soft[0]),
                "image {i}: split call vs alone"
            );
            assert!(alone.soft[0].iter().all(|v| v.is_finite()));
        }
        // 125 columns pool to a row of 62 cells, so x = 124 (cell 62) collides
        // with the next row's first cell: for h rows the cells are
        // 0..=62 + 62 * (h/2 - 1), all occupied.
        assert_eq!(sync.tokens, [1489, 2233, 1241]);
    });
}

#[test]
fn encode_images_refuses_what_it_cannot_run() {
    let cfg = VisionConfig::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 5);
    let st = open("refuse", safetensors_bytes(&w, &[], &[]));
    let pd = cfg.patch_dim() as usize;
    let good = Image::new(4, 4, 4, pd, 31); // P = 20
    with_gpu(|rt| {
        let model = EmbedGemma2VisionModel::load(rt, &st, cfg.clone()).unwrap();
        let ok = model.encode_images(&[good.input()], false).unwrap();
        let mut bad_cases: Vec<(&str, Image)> = Vec::new();
        // Padding not a suffix: a real patch after the first padding patch.
        let mut gap = Image::new(4, 4, 4, pd, 32);
        gap.positions[2 * 17] = 3;
        gap.positions[2 * 17 + 1] = 3;
        bad_cases.push(("a real patch after padding", gap));
        // Only padding.
        bad_cases.push(("all padding", Image::new(0, 0, 4, pd, 33)));
        // P not divisible by k².
        bad_cases.push(("P = 21", Image::new(4, 4, 5, pd, 34)));
        // A position past the table.
        let mut far = Image::new(4, 4, 4, pd, 35);
        far.positions[2 * 3] = 128;
        bad_cases.push(("position 128 of 128", far));
        // A half-padding position (-1 in one coordinate only).
        let mut half = Image::new(4, 4, 4, pd, 36);
        half.positions[2 * 5 + 1] = -1;
        bad_cases.push(("(x, -1)", half));
        // A non-finite pixel in a real patch.
        let mut nan = Image::new(4, 4, 4, pd, 37);
        nan.pixels[7] = f32::NAN;
        bad_cases.push(("NaN pixel", nan));
        // A cell past the output length: P = 20 gives 5 cells, and a patch
        // at x = 127 (inside the table) pools to cell 63.
        let mut cell = Image::new(4, 4, 4, pd, 38);
        cell.positions[0] = 127;
        bad_cases.push(("cell past the output", cell));
        for (what, im) in &bad_cases {
            assert!(model.encode_images(&[im.input()], false).is_err(), "{what}: accepted");
            // A refused image queues nothing and leaves the model usable.
            let again = model.encode_images(&[good.input()], false).unwrap();
            assert_eq!(bits(&again.soft[0]), bits(&ok.soft[0]), "after refusing {what}");
        }
        // A bad image beside a good one refuses the whole call.
        assert!(model
            .encode_images(&[good.input(), bad_cases[0].1.input()], false)
            .is_err());
        // Lengths.
        let short_pixels = ImageInput {
            pixel_values: &good.pixels[..good.pixels.len() - 1],
            position_ids: &good.positions,
        };
        assert!(model.encode_images(&[short_pixels], false).is_err(), "short pixels");
        let odd_positions = ImageInput {
            pixel_values: &good.pixels,
            position_ids: &good.positions[..good.positions.len() - 1],
        };
        assert!(model.encode_images(&[odd_positions], false).is_err(), "odd positions");
        assert!(model.encode_images(&[], false).is_err(), "empty batch");
        let many = vec![good.input(); 257];
        assert!(model.encode_images(&many, false).is_err(), "257 images");
    });
}

#[test]
fn loader_refuses_tensors_it_would_ignore_and_missing_ones() {
    let cfg = VisionConfig::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 2026);
    with_gpu(|rt| {
        for extra in [
            "vision_tower.encoder.layers.1.self_attn.q_proj.linear.input_max",
            "vision_tower.std_bias",
            "embed_vision.embedding_pre_projection_norm.weight",
        ] {
            let st = open("extra", safetensors_bytes(&w, &[(extra, &[4])], &[]));
            let err = EmbedGemma2VisionModel::load(rt, &st, cfg.clone())
                .err()
                .unwrap_or_else(|| panic!("{extra}: loaded"));
            assert!(err.contains(extra), "{extra}: error does not name it: {err}");
        }
        // The text encoder's and audio tower's tensors are not ours.
        let st = open(
            "outside",
            safetensors_bytes(
                &w,
                &[("language_model.norm.weight", &[4]), ("audio_tower.norm.bias", &[4])],
                &[],
            ),
        );
        EmbedGemma2VisionModel::load(rt, &st, cfg.clone()).unwrap();
        for missing in [
            "vision_tower.encoder.layers.1.mlp.down_proj.linear.weight",
            "embed_vision.embedding_projection.weight",
            "vision_tower.patch_embedder.position_embedding_table",
        ] {
            let st = open("missing", safetensors_bytes(&w, &[], &[missing]));
            assert!(
                EmbedGemma2VisionModel::load(rt, &st, cfg.clone()).is_err(),
                "{missing}: loaded without it"
            );
        }
    });
}

/// The checkpoint's vision config, verbatim.
const REAL_CONFIG: &str = r#"{"architectures":["EmbeddingGemma2Model"],"model_type":"embedding_gemma2",
"text_config":{"hidden_size":512,"model_type":"embedding_gemma2_text"},
"vision_config":{"_name_or_path":"","architectures":null,"attention_bias":false,"attention_dropout":0.0,
"default_output_length":280,"dtype":"bfloat16","global_head_dim":64,"head_dim":64,
"hidden_activation":"gelu_pytorch_tanh","hidden_size":768,"initializer_range":0.02,"intermediate_size":3072,
"max_position_embeddings":131072,"model_type":"gemma4_vision","num_attention_heads":12,"num_hidden_layers":16,
"num_key_value_heads":12,"patch_size":16,"pooling_kernel_size":3,"position_embedding_size":10240,
"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":100.0,"rope_type":"axial"},"standardize":false,
"use_clipped_linears":false}}"#;

#[test]
fn config_reads_the_checkpoint() {
    let c = VisionConfig::from_config_json(REAL_CONFIG).unwrap();
    assert_eq!(
        (c.hidden, c.layers, c.heads, c.head_dim, c.intermediate),
        (768, 16, 12, 64, 3072)
    );
    assert_eq!(
        (c.patch_size, c.patch_dim(), c.pooling_kernel, c.position_embedding_size),
        (16, 768, 3, 10240)
    );
    assert_eq!((c.rms_norm_eps, c.rope_theta, c.text_hidden), (1e-6, 100.0, 512));
}

#[test]
fn config_refuses_what_the_forward_does_not_implement() {
    let cases = [
        ("\"gelu_pytorch_tanh\"", "\"silu\""),
        ("\"head_dim\":64", "\"head_dim\":48"),
        ("\"head_dim\":64", "\"head_dim\":0"),
        ("\"num_key_value_heads\":12", "\"num_key_value_heads\":4"),
        ("\"attention_bias\":false", "\"attention_bias\":true"),
        ("\"attention_dropout\":0.0", "\"attention_dropout\":0.1"),
        ("\"standardize\":false", "\"standardize\":true"),
        ("\"use_clipped_linears\":false", "\"use_clipped_linears\":true"),
        ("\"pooling_kernel_size\":3", "\"pooling_kernel_size\":1"),
        ("\"rope_type\":\"axial\"", "\"rope_type\":\"default\""),
        ("\"rope_theta\":100.0", "\"rope_theta\":0.0"),
        ("\"rms_norm_eps\":1e-06", "\"rms_norm_eps\":0"),
        ("\"model_type\":\"gemma4_vision\"", "\"model_type\":\"gemma4_text\""),
        ("\"patch_size\":16", "\"patch_size\":0"),
        ("\"hidden_size\":512", "\"hidden_size\":0"),
        ("\"vision_config\"", "\"vision_cfg\""),
        ("\"text_config\"", "\"text_cfg\""),
    ];
    for (from, to) in cases {
        assert!(REAL_CONFIG.contains(from), "test bug: {from} not in the config");
        let text = REAL_CONFIG.replacen(from, to, 1);
        assert!(
            VisionConfig::from_config_json(&text).is_err(),
            "{from} -> {to}: accepted"
        );
    }
}
