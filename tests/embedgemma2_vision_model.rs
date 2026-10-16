//! EmbeddingGemma 2's vision tower and multimodal embedder against
//! sentence-transformers (transformers 5.19, fp32, eager attention, CPU), on the
//! real checkpoint.
//!
//! Every kernel is checked against an f64 reference on its own
//! (`tests/embedgemma2_vision_kernels.rs`) and the composition on a tiny random
//! checkpoint (`tests/embedgemma2_vision_tiny.rs`); this checks
//! `tessl::embedgemma2_vision` at the checkpoint's shapes: 2520 patches per
//! image, 16 layers of 12 heads, a 3x3 pool, 256 and 264 soft tokens. It needs
//! the real checkpoint and the goldens `tools/embedgemma2_ref/make_mm_reference.py`
//! writes (two synthetic images, 224x224 with 216 padding patches and 640x360
//! with 144, each run alone), so it is opt-in:
//!
//! ```text
//! ~/.venvs/ml/bin/python tools/embedgemma2_ref/make_mm_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//!   cargo test --release --test embedgemma2_vision_model -- --ignored --test-threads=1
//! ```
//!
//! A missing variable or file fails the test rather than skipping it. The
//! processor's own `pixel_values` and `image_position_ids` are the inputs, as
//! the model received them.
//!
//! Bounds, fixed before the first run (the forward is transformers' fp32
//! forward up to operation order, with exact-f32 GEMMs); the observed errors
//! are recorded in `docs/embedgemma2.md`:
//!
//! * the patch embedder's output and the residual stream after every layer,
//!   over the image's real rows (a padding row's output is not meaningful),
//!   within `1e-4` of that tensor's largest magnitude;
//! * the pooled output (after `* sqrt(768)`, padding stripped) within `1e-4` of
//!   its largest magnitude, and the soft tokens (after the embedder) likewise;
//! * both images in one batch equal to each alone, bit for bit;
//! * the soft-token and real-patch counts equal the goldens'.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, VisionConfig};
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;

const REL: f64 = 1e-4;
const LAYERS: usize = 16;
// The bf16 end-to-end bounds (tests/embedgemma2_mm_model.rs's).
const BF16_COS_FLOOR: f64 = 0.9999;
const BF16_VS_HF_COS: f64 = 0.9996;
const BATCH_ABS: f64 = 1e-5;

fn ref_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("EMBEDGEMMA2_MM_REF_DIR")
            .expect("EMBEDGEMMA2_MM_REF_DIR must name the make_mm_reference.py output directory"),
    )
}

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT")
            .expect("EMBEDGEMMA2_SNAPSHOT must name the google/embeddinggemma-2 snapshot directory"),
    )
}

fn npy_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let p = ref_dir().join(name);
    let a = read_npy(&p)
        .unwrap_or_else(|e| panic!("{}: {e} (run tools/embedgemma2_ref/make_mm_reference.py)", p.display()));
    (a.shape.clone(), a.f32_slice().expect("f32 reference").to_vec())
}

fn npy_i32(name: &str) -> Vec<i32> {
    let p = ref_dir().join(name);
    let a = read_npy(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    a.data_i64
        .as_ref()
        .expect("i64 reference")
        .iter()
        .map(|&v| i32::try_from(v).unwrap())
        .collect()
}

/// An image's processor tensors from the goldens.
struct Golden {
    pixels: Vec<f32>,
    positions: Vec<i32>,
    patches: usize,
    real: usize,
}

fn golden(n: usize) -> Golden {
    let (shape, pixels) = npy_f32(&format!("image{n}_pixel_values.npy"));
    let positions = npy_i32(&format!("image{n}_image_position_ids.npy"));
    assert_eq!(shape.len(), 3, "pixel_values shape {shape:?}");
    let patches = shape[1];
    assert_eq!(positions.len(), 2 * patches, "image {n}: position ids vs pixel values");
    let real = (0..patches)
        .position(|t| positions[2 * t] == -1 && positions[2 * t + 1] == -1)
        .unwrap_or(patches);
    Golden {
        pixels,
        positions,
        patches,
        real,
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[ignore]
fn embedgemma2_vision_matches_sentence_transformers() {
    let snap = snapshot();
    let cfg = VisionConfig::from_config_file(&snap.join("config.json")).unwrap();
    let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
    let h = cfg.hidden as usize;
    let text_h = cfg.text_hidden as usize;
    assert_eq!(cfg.layers as usize, LAYERS);
    let goldens = [golden(0), golden(1)];

    with_gpu(|rt| {
        let model = EmbedGemma2VisionModel::load(rt, &st, cfg.clone()).unwrap();
        // Every numeric bound is checked and reported before the test fails,
        // so one run shows the whole picture.
        let mut failures: Vec<String> = Vec::new();
        let mut gate = |name: &str, got: &[f32], want: &[f32]| {
            let scale = want.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
            let err = max_abs(got, want);
            eprintln!(
                "{name}: max abs {err:.3e} = {:.3e} of max |ref| {scale:.3e}",
                err / scale
            );
            if err > REL * scale {
                failures.push(format!("{name}: max abs {err:.3e} > {REL:.0e} * {scale:.3e}"));
            }
        };

        // 1. Each image alone, stage by stage.
        let mut alone: Vec<Vec<f32>> = Vec::new();
        for (n, g) in goldens.iter().enumerate() {
            let input = ImageInput {
                pixel_values: &g.pixels,
                position_ids: &g.positions,
            };
            let out = model.encode_images(&[input], true).unwrap();
            let (pooled_shape, pooled) = npy_f32(&format!("image{n}_pooled.npy"));
            let (soft_shape, soft) = npy_f32(&format!("image{n}_soft.npy"));
            assert_eq!(pooled_shape, [out.tokens[0], h], "image {n}: pooled rows");
            assert_eq!(soft_shape, [out.tokens[0], text_h], "image {n}: soft rows");
            assert_eq!(out.patches[0], g.real, "image {n}: real patches");
            eprintln!(
                "image {n}: {} patches ({} real), {} soft tokens, {} forward(s)",
                g.patches, g.real, out.tokens[0], out.forwards
            );
            let rows = |name: &str| -> Vec<f32> {
                let (shape, all) = npy_f32(name);
                assert_eq!(shape, [g.patches, h], "{name}");
                all[..g.real * h].to_vec()
            };
            gate(
                &format!("image{n} patch_embed"),
                &out.trace.patch_embed[0],
                &rows(&format!("image{n}_patch_embed.npy")),
            );
            assert_eq!(out.trace.layers.len(), LAYERS);
            for l in 0..LAYERS {
                gate(
                    &format!("image{n} layer{l}"),
                    &out.trace.layers[l][0],
                    &rows(&format!("image{n}_layer{l}.npy")),
                );
            }
            gate(&format!("image{n} pooled"), &out.trace.pooled[0], &pooled);
            gate(&format!("image{n} soft"), &out.soft[0], &soft);
            assert!(
                out.soft[0].iter().all(|v| v.is_finite()),
                "image {n}: non-finite soft token"
            );
            alone.push(out.soft.into_iter().next().unwrap());
        }

        // 2. Both in one forward (ragged: 2304 and 2376 real patches), each
        //    equal to itself alone bit for bit.
        let inputs: Vec<ImageInput<'_>> = goldens
            .iter()
            .map(|g| ImageInput {
                pixel_values: &g.pixels,
                position_ids: &g.positions,
            })
            .collect();
        let batched = model.encode_images(&inputs, false).unwrap();
        eprintln!("batched: {} forward(s)", batched.forwards);
        if batched.forwards != 1 {
            failures.push(format!(
                "the two images ran in {} forwards, expected one",
                batched.forwards
            ));
        }
        for (n, a) in alone.iter().enumerate() {
            let same = bits(a) == bits(&batched.soft[n]);
            let drift = max_abs(a, &batched.soft[n]);
            eprintln!("image {n}: alone vs batched max abs {drift:.3e} (bit-identical: {same})");
            if !same {
                failures.push(format!(
                    "image {n}: batched soft tokens differ from alone (max abs {drift:.3e})"
                ));
            }
        }

        // 3. Reversed input order.
        let reversed = [inputs[1], inputs[0]];
        let rev = model.encode_images(&reversed, false).unwrap();
        for (n, a) in alone.iter().enumerate() {
            if bits(&rev.soft[1 - n]) != bits(a) {
                failures.push(format!("image {n}: reversed call differs from alone"));
            }
        }

        assert!(
            failures.is_empty(),
            "{} bound(s) exceeded:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// `GemmOperands::Bf16` on the real checkpoint, the tower alone: each image
/// alone, both in one forward and in the reverse order, bit for bit (the bound
/// is equality: the bf16 GEMM sums K in the same order whatever M is, the
/// padded keys are masked and the padded rows are zero). The tower's own error
/// against the fp32 goldens is reported per stage and not gated here: its gate
/// is the end-to-end embedding below, which is what the tower exists to feed.
#[test]
#[ignore]
fn embedgemma2_vision_bf16_is_batch_independent_and_reports_its_error() {
    let snap = snapshot();
    let cfg = VisionConfig::from_config_file(&snap.join("config.json")).unwrap();
    let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
    let h = cfg.hidden as usize;
    let goldens = [golden(0), golden(1)];
    let inputs: Vec<ImageInput<'_>> = goldens
        .iter()
        .map(|g| ImageInput {
            pixel_values: &g.pixels,
            position_ids: &g.positions,
        })
        .collect();

    with_gpu(|rt| {
        let exact_bytes = EmbedGemma2VisionModel::load(rt, &st, cfg.clone())
            .unwrap()
            .weight_bytes();
        let model = EmbedGemma2VisionModel::load_with(rt, &st, cfg.clone(), GemmOperands::Bf16).unwrap();
        assert_eq!(model.operands(), GemmOperands::Bf16);
        eprintln!(
            "weight bytes: bf16 {} ({:.1} MiB), exact f32 {} ({:.1} MiB), {:.1}%",
            model.weight_bytes(),
            model.weight_bytes() as f64 / (1 << 20) as f64,
            exact_bytes,
            exact_bytes as f64 / (1 << 20) as f64,
            100.0 * model.weight_bytes() as f64 / exact_bytes as f64
        );
        assert!(model.weight_bytes() < exact_bytes);

        let rel = |got: &[f32], want: &[f32]| {
            let scale = want.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
            max_abs(got, want) / scale
        };
        let mut alone: Vec<Vec<f32>> = Vec::new();
        for (n, g) in goldens.iter().enumerate() {
            let out = model.encode_images(&[inputs[n]], true).unwrap();
            let rows = |name: &str| -> Vec<f32> {
                let (_, all) = npy_f32(name);
                all[..g.real * h].to_vec()
            };
            let layers: Vec<String> = (0..LAYERS)
                .map(|l| {
                    format!(
                        "{:.1e}",
                        rel(&out.trace.layers[l][0], &rows(&format!("image{n}_layer{l}.npy")))
                    )
                })
                .collect();
            eprintln!(
                "image {n} (bf16 vs fp32 golden, max abs over max |ref|): patch_embed {:.1e}, layers [{}], pooled {:.1e}, soft {:.1e}",
                rel(&out.trace.patch_embed[0], &rows(&format!("image{n}_patch_embed.npy"))),
                layers.join(" "),
                rel(&out.trace.pooled[0], &npy_f32(&format!("image{n}_pooled.npy")).1),
                rel(&out.soft[0], &npy_f32(&format!("image{n}_soft.npy")).1),
            );
            assert!(
                out.soft[0].iter().all(|v| v.is_finite()),
                "image {n}: non-finite soft token"
            );
            alone.push(out.soft.into_iter().next().unwrap());
        }

        let batched = model.encode_images(&inputs, false).unwrap();
        assert_eq!(batched.forwards, 1, "the two images must share one forward");
        let rev = model.encode_images(&[inputs[1], inputs[0]], false).unwrap();
        for (n, a) in alone.iter().enumerate() {
            assert_eq!(
                bits(a),
                bits(&batched.soft[n]),
                "image {n}: batched differs from alone (max abs {:.3e})",
                max_abs(a, &batched.soft[n])
            );
            assert_eq!(
                bits(a),
                bits(&rev.soft[1 - n]),
                "image {n}: reversed call differs from alone"
            );
        }
    });
}

/// The image embeddings end to end with both models in bf16 (`GemmOperands::Bf16`
/// through the facade: the tower and the text encoder; the audio tower is not
/// in these calls), against sentence-transformers' fp32 embeddings and its own
/// bf16 forward, the pattern `tests/embedgemma2_mm_model.rs` holds the text
/// encoder to. Bounds, fixed before the first run (that test's):
///
/// * each image at cosine `0.9999` or better against `image{n}_emb_f32.npy`;
/// * each image at cosine `0.9996` or better against `image{n}_emb_bf16.npy`
///   (the triangle inequality over two forwards each within `1 - cos <= 1e-4`
///   of fp32);
/// * over the two images the mean of `1 - cos` against fp32 at most that of
///   transformers' bf16 forward (`emb_bf16.npy` against `emb_f32.npy`);
/// * each image alone within `1e-5` max abs of itself inside one call of both
///   (the text encoder's bound; the tower alone is held to bit equality by the
///   test above, and whether the embeddings are bit-identical is reported).
#[test]
#[ignore]
fn embedgemma2_multimodal_bf16_images_are_no_worse_than_transformers_bf16() {
    let snap = snapshot();
    let goldens = [golden(0), golden(1)];
    let emb = |n: usize, kind: &str| npy_f32(&format!("image{n}_emb_{kind}.npy")).1;
    let (f32_ref, bf16_ref): (Vec<_>, Vec<_>) = (0..2).map(|n| (emb(n, "f32"), emb(n, "bf16"))).unzip();
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snap, GemmOperands::Bf16).unwrap();
        assert_eq!(m.text().operands(), GemmOperands::Bf16);
        assert_eq!(m.vision().operands(), GemmOperands::Bf16);
        let item = |n: usize| {
            MmItem::Image(ImageInput {
                pixel_values: &goldens[n].pixels,
                position_ids: &goldens[n].positions,
            })
        };
        let dim = f32_ref[0].len();
        let mut failures: Vec<String> = Vec::new();
        let (mut deficit_ours, mut deficit_hf) = (0.0f64, 0.0f64);
        let mut alone: Vec<Vec<f32>> = Vec::new();
        for n in 0..2 {
            let e = m.encode(&[item(n)], None).unwrap().embeddings;
            let (cos_f, cos_h, cos_hf) = (
                cosine(&e, &f32_ref[n]),
                cosine(&e, &bf16_ref[n]),
                cosine(&bf16_ref[n], &f32_ref[n]),
            );
            eprintln!(
                "image {n}: tessl bf16 (tower + text) vs fp32 cosine {cos_f:.8} (max abs {:.2e}); vs transformers bf16 \
                 cosine {cos_h:.8}; transformers bf16 vs fp32 cosine {cos_hf:.8}",
                max_abs(&e, &f32_ref[n])
            );
            deficit_ours += 1.0 - cos_f;
            deficit_hf += 1.0 - cos_hf;
            if cos_f < BF16_COS_FLOOR {
                failures.push(format!("image {n}: cosine vs fp32 {cos_f:.8} < {BF16_COS_FLOOR}"));
            }
            if cos_h < BF16_VS_HF_COS {
                failures.push(format!(
                    "image {n}: cosine vs transformers bf16 {cos_h:.8} < {BF16_VS_HF_COS}"
                ));
            }
            let norm = e.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "image {n}: norm {norm}");
            alone.push(e);
        }
        eprintln!(
            "mean 1 - cos vs fp32: tessl bf16 {:.3e}, transformers bf16 {:.3e}",
            deficit_ours / 2.0,
            deficit_hf / 2.0
        );
        if deficit_ours > deficit_hf {
            failures.push(format!(
                "tessl bf16 is further from fp32 (mean 1 - cos {:.3e}) than transformers' bf16 forward ({:.3e})",
                deficit_ours / 2.0,
                deficit_hf / 2.0
            ));
        }
        let both = m.encode(&[item(0), item(1)], None).unwrap().embeddings;
        for (n, a) in alone.iter().enumerate() {
            let drift = max_abs(a, &both[n * dim..(n + 1) * dim]);
            eprintln!(
                "image {n}: alone vs both in one call max abs {drift:.3e} (bit-identical: {})",
                bits(a) == bits(&both[n * dim..(n + 1) * dim])
            );
            if drift > BATCH_ABS {
                failures.push(format!(
                    "image {n}: alone vs inside one call max abs {drift:.3e} > {BATCH_ABS:.0e}"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} bound(s) exceeded:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}
