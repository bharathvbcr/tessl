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
//!
//! `GemmOperands::Bf16` (`load_with`: the projections stay bf16 on the device,
//! every GEMM input is rounded to bf16, the attention runs on bf16 tiles;
//! accumulation, the residual stream, the norms, the rotation, the pooling and
//! the soft tokens stay f32) runs the same checks, on a config with two heads
//! of 64 (the bf16 attention is not compiled for 32) and a checkpoint whose
//! every projection is stored bf16, against the same f64 forward over exactly
//! those weights, so only the activations round. Its bounds are derived from
//! the rounding, as `tests/embedgemma2_tiny.rs` derives the text encoder's,
//! and fixed before the first run:
//!
//! * a bf16 rounding is a relative error of at most `u = 2^-8` (3.9e-3) and
//!   about `u / sqrt(3)` (2.3e-3) rms. A GEMM output sums K rounded terms that
//!   add as a random walk, as the terms do, so its error is about 2.3e-3 of
//!   its own rms whatever K is, and a maximum over a few thousand elements sits
//!   near 4.5 rms: about 1e-2 of a rounded-input GEMM's rms;
//! * the patch embedder rounds the scaled pixels (the weights are exact) and
//!   adds the position embedding exactly: one rounded-input GEMM, so about 1e-2
//!   of the output's rms and less of its maximum. The bound is `2e-2`;
//! * a layer's attention branch rounds its normed input, q, k and v, the
//!   probability tile that multiplies V and the attention output, five sources
//!   at 2.3e-3 each plus the softmax's response to rounded scores (scale 1.0,
//!   scores of order 1): about 6e-3 of the branch rms, then `rms_norm` makes
//!   the branch scale-free and O(1) per element; the MLP branch rounds its
//!   input and the gated activation (two sources, 3e-3, plus the GELU's slope
//!   of order 1). Two layers of two branches add in quadrature to about twice
//!   one branch, 1.2e-2 of a branch's scale at the rms and 5e-2 at a maximum
//!   over thousands of elements, against a residual whose largest element is
//!   several times its rms (the branches add up): the bound for each layer's
//!   residual stream and for the pooled output (an average of nine rows, which
//!   cannot be worse than the rows) is `5e-2` of its largest magnitude, the
//!   text encoder's, still far below a wrong layer, rope axis or pooling cell
//!   (O(1));
//! * a soft token is the pooled row, a weightless `rms_norm` and one more
//!   rounded-input GEMM: its direction moves by about the pooled row's
//!   relative error (a few 1e-3), so `1 - cos` is about 1e-5 and a worst case
//!   over the 19 tokens of the four images near 1e-4. The bounds are cosine
//!   `>= 0.999` per token (10x of margin) and `5e-2` of the largest soft
//!   value as max abs; an empty pooling cell is exactly zero in both;
//! * alone vs batched, the reversed call, the poisoned activations and the
//!   split call stay bit-identical: the bf16 GEMM sums K in the same order
//!   whatever M is, so the f32 values that round are the same bits.

mod common;

use common::embedgemma2::{pack_safetensors, RawTensor};
use common::embedgemma2_mm::{vision_weights, TowerWeight, VisionStorage};
use common::embedgemma2_vision::{vision_forward, VisionDims, VisionRef};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, VisionConfig, MAX_BATCH_PATCHES};
use tessl::gemm::GemmOperands;
use tessl::safetensors::SafeTensors;

const LAYER_REL: f64 = 1e-4;
const SOFT_ABS: f64 = 1e-4;

// GemmOperands::Bf16 bounds, derived in the module docs and fixed before the
// first run.
const BF16_PATCH_REL: f64 = 2e-2;
const BF16_LAYER_REL: f64 = 5e-2;
const BF16_SOFT_REL: f64 = 5e-2;
const BF16_TOKEN_COS: f64 = 0.999;

const CONFIG: &str = r#"{"model_type":"embedding_gemma2",
"text_config":{"hidden_size":48,"model_type":"embedding_gemma2_text"},
"vision_config":{"model_type":"gemma4_vision","hidden_activation":"gelu_pytorch_tanh","hidden_size":96,
"num_hidden_layers":2,"num_attention_heads":3,"num_key_value_heads":3,"head_dim":32,"intermediate_size":160,
"patch_size":4,"pooling_kernel_size":2,"position_embedding_size":128,"rms_norm_eps":1e-06,
"attention_bias":false,"attention_dropout":0.0,"standardize":false,"use_clipped_linears":false,
"rope_parameters":{"rope_theta":100.0,"rope_type":"axial"}}}"#;

/// The tiny checkpoint's tensors (`common::embedgemma2_mm::vision_weights`),
/// with the values the device reads for the host reference.
struct Weights(Vec<TowerWeight>);

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

/// A random checkpoint. The patch projection and the position table are bf16, as the
/// real one stores them; with `all_bf16` so is every projection (the real
/// checkpoint stores them all bf16), which `GemmOperands::Bf16` needs. The
/// norm weights are f32 in both.
fn random_weights(cfg: &VisionConfig, seed: u64, all_bf16: bool) -> Weights {
    Weights(vision_weights(
        cfg,
        seed,
        VisionStorage {
            projections_bf16: all_bf16,
            norms_bf16: false,
        },
    ))
}

/// safetensors bytes: every weight, plus `extra` tensors named as given (f32
/// zeros of the given shape), minus the ones named in `skip`.
fn safetensors_bytes(w: &Weights, extra: &[(&str, &[usize])], skip: &[&str]) -> Vec<u8> {
    let mut tensors: Vec<RawTensor> =
        w.0.iter()
            .filter(|t| !skip.contains(&t.name.as_str()))
            .map(TowerWeight::raw)
            .collect();
    for &(name, shape) in extra {
        let n: usize = shape.iter().product();
        tensors.push(RawTensor {
            name: name.to_string(),
            dtype: "F32",
            shape: shape.to_vec(),
            bytes: vec![0u8; n * 4],
        });
    }
    pack_safetensors(&tensors)
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

/// FNV-1a over the bits of every value of a traced call, in a fixed order.
fn output_hash(out: &tessl::embedgemma2_vision::VisionOutput) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |v: &[f32]| {
        for x in v {
            for byte in x.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    };
    for s in &out.soft {
        feed(s);
    }
    for p in &out.trace.patch_embed {
        feed(p);
    }
    for layer in &out.trace.layers {
        for l in layer {
            feed(l);
        }
    }
    for p in &out.trace.pooled {
        feed(p);
    }
    hash
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

/// The bounds one mode is held to (see the module docs for where they come from).
struct Bounds {
    /// The patch embedder's output, each layer's residual stream and the
    /// pooled output, each within this of its own largest magnitude.
    patch_rel: f64,
    layer_rel: f64,
    /// Soft tokens: max abs (exact f32), or within `soft_rel` of the largest
    /// magnitude and each token at cosine `token_cos` or better (bf16).
    soft: SoftBound,
}

enum SoftBound {
    Abs(f64),
    RelAndCosine { rel: f64, cos: f64 },
}

fn bounds(operands: GemmOperands) -> Bounds {
    match operands {
        GemmOperands::ExactF32 => Bounds {
            patch_rel: LAYER_REL,
            layer_rel: LAYER_REL,
            soft: SoftBound::Abs(SOFT_ABS),
        },
        GemmOperands::Bf16 => Bounds {
            patch_rel: BF16_PATCH_REL,
            layer_rel: BF16_LAYER_REL,
            soft: SoftBound::RelAndCosine {
                rel: BF16_SOFT_REL,
                cos: BF16_TOKEN_COS,
            },
        },
    }
}

/// The mode's config and checkpoint: the exact-f32 test's, or for bf16 a
/// config with two heads of 64 (the bf16 attention is compiled for head dim 64,
/// 256 and 512) and a checkpoint whose every projection is stored bf16, so the
/// f64 forward uses exactly the weights the device holds and only the
/// activations round.
fn setup(operands: GemmOperands, seed: u64) -> (VisionConfig, Weights) {
    let (text, all_bf16) = match operands {
        GemmOperands::ExactF32 => (CONFIG.to_string(), false),
        GemmOperands::Bf16 => {
            let from = "\"num_attention_heads\":3,\"num_key_value_heads\":3,\"head_dim\":32";
            assert!(CONFIG.contains(from), "test bug: the config changed");
            (
                CONFIG.replace(
                    from,
                    "\"num_attention_heads\":2,\"num_key_value_heads\":2,\"head_dim\":64",
                ),
                true,
            )
        }
    };
    let cfg = VisionConfig::from_config_json(&text).unwrap();
    let w = random_weights(&cfg, seed, all_bf16);
    (cfg, w)
}

fn check_against_reference(
    tag: &str,
    out: &tessl::embedgemma2_vision::VisionOutput,
    i: usize,
    want: &VisionRef,
    b: &Bounds,
) {
    assert_eq!(out.patches[i], want.n, "{tag}: real patches");
    assert_eq!(out.tokens[i], want.tokens, "{tag}: soft tokens");
    let t = &out.trace;
    check_rel(
        &format!("{tag} patch_embed"),
        &t.patch_embed[i],
        &want.patch_embed,
        b.patch_rel,
    );
    for (l, w) in want.layers.iter().enumerate() {
        check_rel(&format!("{tag} layer {l}"), &t.layers[l][i], w, b.layer_rel);
    }
    check_rel(&format!("{tag} pooled"), &t.pooled[i], &want.pooled, b.layer_rel);
    match b.soft {
        SoftBound::Abs(abs) => {
            let err = max_abs(&out.soft[i], &want.soft);
            eprintln!("{tag} soft: max abs {err:.3e}");
            assert!(err <= abs, "{tag} soft: max abs {err:.3e} > {abs:.0e}");
        }
        SoftBound::RelAndCosine { rel, cos } => {
            check_rel(&format!("{tag} soft"), &out.soft[i], &want.soft, rel);
            let width = want.soft.len() / want.tokens;
            let mut worst: f64 = 1.0;
            for (tok, (g, w)) in out.soft[i].chunks(width).zip(want.soft.chunks(width)).enumerate() {
                let wn = w.iter().map(|v| v * v).sum::<f64>().sqrt();
                let gn = g.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
                if wn == 0.0 {
                    // An empty cell: exactly +0 in the reference, so exactly 0 here.
                    assert!(
                        gn == 0.0,
                        "{tag} soft token {tok}: reference is zero, got norm {gn:.3e}"
                    );
                    continue;
                }
                let dot: f64 = g.iter().zip(w).map(|(x, y)| f64::from(*x) * y).sum();
                worst = worst.min(dot / (gn * wn));
            }
            eprintln!("{tag} soft: worst token cosine {worst:.8}");
            assert!(worst >= cos, "{tag} soft: worst token cosine {worst:.8} < {cos}");
        }
    }
}

#[test]
fn tiny_vision_matches_the_f64_forward_alone_and_batched() {
    matches_the_f64_forward(GemmOperands::ExactF32);
}

/// The same checks at bf16 operands, against the f64 forward over the same
/// bf16 weights, under the derived bounds in the module docs.
#[test]
fn tiny_vision_bf16_matches_the_f64_forward_alone_and_batched() {
    matches_the_f64_forward(GemmOperands::Bf16);
}

fn matches_the_f64_forward(operands: GemmOperands) {
    let b = bounds(operands);
    let (cfg, w) = setup(operands, 2026);
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
        let model = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), operands).unwrap();
        assert_eq!(model.operands(), operands);

        // 1. Each image alone, layer by layer.
        let mut alone = Vec::new();
        for (i, im) in imgs.iter().enumerate() {
            let out = model.encode_images(&[im.input()], true).unwrap();
            assert_eq!(out.forwards, 1);
            assert_eq!(out.trace.layers.len(), cfg.layers as usize);
            check_against_reference(&format!("image {i} alone"), &out, 0, &refs[i], &b);
            assert_eq!(out.soft[0].len(), out.tokens[0] * cfg.text_hidden as usize);
            alone.push(out.soft.into_iter().next().unwrap());
        }

        // 2. All four in one forward (ragged: 48, 20, 1, 4 real patches),
        //    against the f64 forward and against themselves alone, bit for bit.
        let all: Vec<ImageInput<'_>> = imgs.iter().map(Image::input).collect();
        let batched = model.encode_images(&all, true).unwrap();
        assert_eq!(batched.forwards, 1, "the tiny batch must share one forward");
        // One number for the whole output (soft tokens, patch embedder, every
        // layer, pooled), to compare across builds with `--nocapture`: equal
        // means bit-identical.
        eprintln!("HASH tiny_vision {operands:?} {:016x}", output_hash(&batched));
        for i in 0..imgs.len() {
            check_against_reference(&format!("image {i} batched"), &batched, i, &refs[i], &b);
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
    poisoned_activations_change_nothing(GemmOperands::ExactF32);
}

#[test]
fn poisoned_unzeroed_activations_change_nothing_at_bf16() {
    poisoned_activations_change_nothing(GemmOperands::Bf16);
}

fn poisoned_activations_change_nothing(operands: GemmOperands) {
    let (cfg, w) = setup(operands, 7);
    let st = open("poison", safetensors_bytes(&w, &[], &[]));
    let imgs = images(cfg.patch_dim() as usize);
    let all: Vec<ImageInput<'_>> = imgs.iter().map(Image::input).collect();
    with_gpu(|rt| {
        let model = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), operands).unwrap();
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
    a_split_call_matches_each_image_alone(GemmOperands::ExactF32);
}

#[test]
fn a_call_that_splits_into_forwards_matches_each_image_alone_at_bf16() {
    a_split_call_matches_each_image_alone(GemmOperands::Bf16);
}

fn a_split_call_matches_each_image_alone(operands: GemmOperands) {
    let (cfg, w) = setup(operands, 11);
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
        let model = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), operands).unwrap();
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
    refuses_what_it_cannot_run(GemmOperands::ExactF32);
}

#[test]
fn encode_images_refuses_what_it_cannot_run_at_bf16() {
    refuses_what_it_cannot_run(GemmOperands::Bf16);
}

fn refuses_what_it_cannot_run(operands: GemmOperands) {
    let (cfg, w) = setup(operands, 5);
    let st = open("refuse", safetensors_bytes(&w, &[], &[]));
    let pd = cfg.patch_dim() as usize;
    let good = Image::new(4, 4, 4, pd, 31); // P = 20
    with_gpu(|rt| {
        let model = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), operands).unwrap();
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
        // More real patches than one forward may hold (the limit is checked
        // before the positions, which here also run past the 128-row table).
        let too_many = Image::new(128, 129, 0, pd, 39);
        let err = model.encode_images(&[too_many.input()], false).unwrap_err();
        assert!(err.contains("the limit is 16384"), "{err}");
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
    let w = random_weights(&cfg, 2026, false);
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

/// What `GemmOperands::Bf16` refuses at load, and what it holds on the device:
/// a head dim the bf16 attention is not compiled for, a projection the
/// checkpoint stores in another dtype (named, not rounded), and bytes that are
/// exactly the projections at two bytes an element where exact f32 holds four.
#[test]
fn bf16_load_refuses_what_it_cannot_run_and_holds_the_weights_at_two_bytes() {
    let (cfg32, w32) = setup(GemmOperands::ExactF32, 3);
    let st32 = open("heads_of_32", safetensors_bytes(&w32, &[], &[]));
    let (cfg, w) = setup(GemmOperands::Bf16, 3);
    let st = open("all_bf16", safetensors_bytes(&w, &[], &[]));
    // The bf16 config over a checkpoint whose projections are stored f32.
    let w_f32 = random_weights(&cfg, 3, false);
    let st_f32 = open("f32_projections", safetensors_bytes(&w_f32, &[], &[]));

    with_gpu(|rt| {
        let err = EmbedGemma2VisionModel::load_with(rt, &st32, cfg32.clone(), GemmOperands::Bf16)
            .err()
            .expect("bf16 operands loaded a head dim of 32");
        assert!(err.contains("head dim 32"), "{err}");
        // Exact f32 still loads it.
        EmbedGemma2VisionModel::load_with(rt, &st32, cfg32.clone(), GemmOperands::ExactF32).unwrap();

        let err = EmbedGemma2VisionModel::load_with(rt, &st_f32, cfg.clone(), GemmOperands::Bf16)
            .err()
            .expect("bf16 operands rounded an f32 projection");
        assert!(err.contains("q_proj"), "the error does not name the tensor: {err}");

        // Weight bytes: every projection at 2 (bf16) or 4 (f32) bytes an
        // element, the position table bf16 under both, the norms f32 under both.
        let exact = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), GemmOperands::ExactF32).unwrap();
        let bf16 = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), GemmOperands::Bf16).unwrap();
        assert_eq!(
            (exact.operands(), bf16.operands()),
            (GemmOperands::ExactF32, GemmOperands::Bf16)
        );
        let d = dims(&cfg);
        let qw = d.heads * d.head_dim;
        let per_layer = d.hidden * 3 * qw + qw * d.hidden + 2 * d.hidden * d.inter + d.inter * d.hidden;
        let projection_elems = d.layers * per_layer + d.patch_dim * d.hidden + d.hidden * d.text_hidden;
        let table = 2 * d.table_len * d.hidden * 2;
        let norms = 4 * (d.layers * (4 * d.hidden + 2 * d.head_dim) + d.hidden);
        assert_eq!(exact.weight_bytes(), 4 * projection_elems + table + norms);
        assert_eq!(bf16.weight_bytes(), 2 * projection_elems + table + norms);
        assert!(bf16.weight_bytes() < exact.weight_bytes());
    });
}

/// The exact-f32 forward needs the runtime's relaxed (tf32-class) f32 GEMMs
/// off and says so; the bf16 forward runs no f32 GEMM, so the switch does not
/// touch it and a result with it on is the same bits.
#[test]
fn relaxed_precision_is_refused_by_exact_f32_and_moves_nothing_at_bf16() {
    for operands in [GemmOperands::ExactF32, GemmOperands::Bf16] {
        let (cfg, w) = setup(operands, 13);
        let st = open("relaxed", safetensors_bytes(&w, &[], &[]));
        let img = Image::new(6, 4, 8, cfg.patch_dim() as usize, 41);
        with_gpu(|rt| {
            let model = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), operands).unwrap();
            let strict = model.encode_images(&[img.input()], false).unwrap();
            rt.set_relaxed_precision(true);
            let relaxed = model.encode_images(&[img.input()], false);
            rt.set_relaxed_precision(false);
            match operands {
                GemmOperands::ExactF32 => {
                    let Err(err) = relaxed else {
                        panic!("ExactF32 ran with relaxed precision on");
                    };
                    assert!(err.contains("exact-f32"), "{err}");
                }
                GemmOperands::Bf16 => assert_eq!(
                    bits(&relaxed.expect("Bf16 refused relaxed precision").soft[0]),
                    bits(&strict.soft[0]),
                    "relaxed precision changed a bf16 soft token"
                ),
            }
        });
    }
}

/// The bf16 forward's operands really are rounded: it is not the exact-f32
/// forward in disguise (its soft tokens differ from the exact ones over the
/// same weights), and it is not so far from it as to be a different model.
#[test]
fn bf16_rounds_its_operands_and_stays_near_the_exact_forward() {
    let (cfg, w) = setup(GemmOperands::Bf16, 17);
    let st = open("rounds", safetensors_bytes(&w, &[], &[]));
    let img = Image::new(8, 6, 16, cfg.patch_dim() as usize, 51);
    with_gpu(|rt| {
        let exact = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), GemmOperands::ExactF32).unwrap();
        let bf16 = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), GemmOperands::Bf16).unwrap();
        let (e, b) = (
            exact.encode_images(&[img.input()], false).unwrap(),
            bf16.encode_images(&[img.input()], false).unwrap(),
        );
        assert_ne!(bits(&e.soft[0]), bits(&b.soft[0]), "bf16 operands changed nothing");
        let want: Vec<f64> = e.soft[0].iter().map(|&v| f64::from(v)).collect();
        check_rel("bf16 vs exact soft tokens", &b.soft[0], &want, BF16_SOFT_REL);
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
