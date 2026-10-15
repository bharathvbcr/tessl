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
use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, VisionConfig};
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;

const REL: f64 = 1e-4;
const LAYERS: usize = 16;

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
