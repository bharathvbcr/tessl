//! EmbeddingGemma 2's multimodal merge (`EmbedGemma2Model::encode_multimodal`)
//! against sentence-transformers' image and audio embeddings, from the
//! towers' golden soft tokens, so neither tower is needed. Opt-in, like
//! `tests/embedgemma2_model.rs`:
//!
//! ```text
//! ~/.venvs/ml/bin/python tools/embedgemma2_ref/make_mm_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   EMBEDGEMMA2_MM_REF_DIR=target/embedgemma2_mm_ref \
//!   cargo test --release --test embedgemma2_mm_model -- --ignored --test-threads=1
//! ```
//!
//! `EMBEDGEMMA2_MM_REF_DIR` overrides `target/embedgemma2_mm_ref`. A missing
//! variable or file fails the test rather than skipping it.
//!
//! For each of `image0_`, `image1_`, `audio0_`, `audio1_` the generator saves
//! the processor's `ids.npy` (placeholders included), the soft tokens in
//! scatter order (`soft.npy`, `[S, 512]`), the text model's output after its
//! final norm (`lm_final.npy`) and sentence-transformers' embedding in fp32
//! and with bf16 weights. Bounds, fixed before the first run, the text
//! tests' (`tests/embedgemma2_model.rs`):
//!
//! `ExactF32` (transformers' fp32 forward up to operation order):
//!
//! * `lm_final` (the trace's last entry, every token) within `1e-4` of its
//!   largest magnitude;
//! * the embedding within `1e-4` max abs of `emb_f32.npy`, cosine at least
//!   `0.99999`;
//! * the ids [`EmbedGemma2Config::image_sequence`] and
//!   [`EmbedGemma2Config::audio_sequence`] build equal `ids.npy`;
//! * each input alone and inside one batch of all four within `1e-5`.
//!
//! `Bf16`, per input: cosine at least `0.9999` against the fp32 embedding
//! and `0.9996` against `emb_bf16.npy` (the triangle inequality over two
//! forwards each within `1 - cos <= 1e-4` of fp32); over the four inputs the
//! mean of `1 - cos` against fp32 at most that of `emb_bf16.npy`; alone vs
//! batched within `1e-5`.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, MmSequence};
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;

const PREFIX: &str = "language_model.";
const LM_REL: f64 = 1e-4;
const EMB_ABS: f64 = 1e-4;
const EMB_COS: f64 = 0.99999;
const BATCH_ABS: f64 = 1e-5;
const BF16_COS_FLOOR: f64 = 0.9999;
const BF16_VS_HF_COS: f64 = 0.9996;
/// `make_mm_reference.py`'s inputs, in the order this test reports them.
const INPUTS: [&str; 4] = ["image0_", "image1_", "audio0_", "audio1_"];

fn ref_dir() -> PathBuf {
    std::env::var_os("EMBEDGEMMA2_MM_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/embedgemma2_mm_ref"))
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

fn npy_ids(name: &str) -> Vec<u32> {
    let p = ref_dir().join(name);
    let a = read_npy(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    a.data_i64
        .clone()
        .expect("i64 reference")
        .into_iter()
        .map(|i| u32::try_from(i).unwrap())
        .collect()
}

/// One golden input.
struct Input {
    name: &'static str,
    ids: Vec<u32>,
    soft: Vec<f32>,
    lm_final: Vec<f32>,
    emb_f32: Vec<f32>,
    emb_bf16: Vec<f32>,
}

fn inputs(cfg: &EmbedGemma2Config) -> Vec<Input> {
    let (h, e) = (cfg.hidden as usize, cfg.embedding_dim as usize);
    INPUTS
        .iter()
        .map(|&name| {
            let ids = npy_ids(&format!("{name}ids.npy"));
            let (soft_shape, soft) = npy_f32(&format!("{name}soft.npy"));
            let (lm_shape, lm_final) = npy_f32(&format!("{name}lm_final.npy"));
            assert_eq!(soft_shape.len(), 2);
            assert_eq!(soft_shape[1], h, "{name} soft width");
            assert_eq!(lm_shape, [ids.len(), h], "{name} lm_final");
            let tm = cfg.special_tokens.multimodal(name).unwrap();
            let placeholders = ids.iter().filter(|&&i| tm.is_placeholder(i)).count();
            assert_eq!(placeholders, soft_shape[0], "{name}: placeholders vs soft rows");
            // The builders reproduce the processor's layout.
            let built = if name.starts_with("image") {
                cfg.image_sequence(soft_shape[0])
            } else {
                cfg.audio_sequence(soft_shape[0])
            };
            assert_eq!(built.unwrap(), ids, "{name}: built ids vs ids.npy");
            let (s32, emb_f32) = npy_f32(&format!("{name}emb_f32.npy"));
            let (s16, emb_bf16) = npy_f32(&format!("{name}emb_bf16.npy"));
            assert_eq!((s32, s16), (vec![e], vec![e]), "{name} embeddings");
            Input {
                name,
                ids,
                soft,
                lm_final,
                emb_f32,
                emb_bf16,
            }
        })
        .collect()
}

impl Input {
    fn seq(&self) -> MmSequence<'_> {
        MmSequence {
            ids: &self.ids,
            soft: &self.soft,
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

fn finish(failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{} bound(s) exceeded:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore]
fn embedgemma2_multimodal_matches_sentence_transformers() {
    let snap = snapshot();
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json")).unwrap();
    let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
    let inputs = inputs(&cfg);
    let (h, dim) = (cfg.hidden as usize, cfg.embedding_dim as usize);

    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        let mut failures: Vec<String> = Vec::new();

        // 1. Alone: lm_final from the trace, then the embedding.
        let mut alone = Vec::new();
        for inp in &inputs {
            let t = inp.ids.len();
            let out = model.encode_multimodal(&[inp.seq()], None, true).unwrap();
            let got = out.trace.last().expect("trace");
            let scale = inp.lm_final.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
            let err = max_abs(&got[..t * h], &inp.lm_final);
            eprintln!(
                "{}: {t} tokens ({} soft), lm_final max abs {err:.3e} = {:.3e} of max |ref| {scale:.3e}",
                inp.name,
                inp.soft.len() / h,
                err / scale
            );
            if err > LM_REL * scale {
                failures.push(format!(
                    "{}: lm_final max abs {err:.3e} > {LM_REL:.0e} * {scale:.3e}",
                    inp.name
                ));
            }
            let e = out.embeddings;
            let (err, cos) = (max_abs(&e, &inp.emb_f32), cosine(&e, &inp.emb_f32));
            let cos16 = cosine(&inp.emb_bf16, &inp.emb_f32);
            eprintln!(
                "{}: embedding max abs {err:.2e}, cosine {cos:.8}; bf16 ref vs fp32 ref cosine {cos16:.6}",
                inp.name
            );
            if !(err <= EMB_ABS && cos >= EMB_COS) {
                failures.push(format!("{}: embedding max abs {err:.3e}, cosine {cos:.8}", inp.name));
            }
            let norm = e.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "{}: norm {norm}", inp.name);
            alone.push(e);
        }

        // 2. All four in one call.
        let seqs: Vec<MmSequence<'_>> = inputs.iter().map(Input::seq).collect();
        let out = model.encode_multimodal(&seqs, None, false).unwrap();
        eprintln!("batch of {}: {} forward(s)", seqs.len(), out.forwards);
        for (n, (inp, e)) in inputs.iter().zip(&alone).enumerate() {
            let got = &out.embeddings[n * dim..(n + 1) * dim];
            let (err, cos) = (max_abs(got, &inp.emb_f32), cosine(got, &inp.emb_f32));
            let drift = max_abs(e, got);
            eprintln!(
                "{}: batched max abs {err:.2e}, cosine {cos:.8}, alone vs batched max abs {drift:.2e}",
                inp.name
            );
            if !(err <= EMB_ABS && cos >= EMB_COS) {
                failures.push(format!("{}: batched max abs {err:.3e}, cosine {cos:.8}", inp.name));
            }
            if drift > BATCH_ABS {
                failures.push(format!("{}: alone vs batched max abs {drift:.3e}", inp.name));
            }
        }

        // 3. The Matryoshka prefix of a soft sequence is its own embedding's
        //    prefix, renormalized.
        let d = 256usize;
        let pre = model
            .encode_multimodal(&seqs, Some(d as u32), false)
            .unwrap()
            .embeddings;
        for (n, inp) in inputs.iter().enumerate() {
            let mut want: Vec<f32> = inp.emb_f32[..d].to_vec();
            let norm = want.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt() as f32;
            want.iter_mut().for_each(|v| *v /= norm);
            let got = &pre[n * d..(n + 1) * d];
            let (err, cos) = (max_abs(got, &want), cosine(got, &want));
            if !(err <= EMB_ABS && cos >= EMB_COS) {
                failures.push(format!("{}: prefix {d} max abs {err:.3e}, cosine {cos:.8}", inp.name));
            }
        }
        finish(failures);
    });
}

#[test]
#[ignore]
fn embedgemma2_multimodal_bf16_is_no_worse_than_transformers_bf16() {
    let snap = snapshot();
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json")).unwrap();
    let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
    let inputs = inputs(&cfg);
    let dim = cfg.embedding_dim as usize;

    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), GemmOperands::Bf16).unwrap();
        let mut failures: Vec<String> = Vec::new();
        let (mut deficit_ours, mut deficit_hf) = (0.0f64, 0.0f64);
        let mut alone = Vec::new();
        for inp in &inputs {
            let e = model.encode_multimodal(&[inp.seq()], None, false).unwrap().embeddings;
            let (cos_f, cos_h, cos_hf) = (
                cosine(&e, &inp.emb_f32),
                cosine(&e, &inp.emb_bf16),
                cosine(&inp.emb_bf16, &inp.emb_f32),
            );
            eprintln!(
                "{}: tessl bf16 vs fp32 cosine {cos_f:.8} (max abs {:.2e}); vs transformers bf16 cosine {cos_h:.8} \
                 (max abs {:.2e}); transformers bf16 vs fp32 cosine {cos_hf:.8}",
                inp.name,
                max_abs(&e, &inp.emb_f32),
                max_abs(&e, &inp.emb_bf16)
            );
            deficit_ours += 1.0 - cos_f;
            deficit_hf += 1.0 - cos_hf;
            if cos_f < BF16_COS_FLOOR {
                failures.push(format!("{}: cosine vs fp32 {cos_f:.8} < {BF16_COS_FLOOR}", inp.name));
            }
            if cos_h < BF16_VS_HF_COS {
                failures.push(format!(
                    "{}: cosine vs transformers bf16 {cos_h:.8} < {BF16_VS_HF_COS}",
                    inp.name
                ));
            }
            let norm = e.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "{}: norm {norm}", inp.name);
            alone.push(e);
        }
        let k = inputs.len() as f64;
        eprintln!(
            "mean 1 - cos vs fp32: tessl bf16 {:.3e}, transformers bf16 {:.3e}",
            deficit_ours / k,
            deficit_hf / k
        );
        if deficit_ours > deficit_hf {
            failures.push(format!(
                "tessl bf16 is further from fp32 (mean 1 - cos {:.3e}) than transformers' bf16 forward ({:.3e})",
                deficit_ours / k,
                deficit_hf / k
            ));
        }

        let seqs: Vec<MmSequence<'_>> = inputs.iter().map(Input::seq).collect();
        let out = model.encode_multimodal(&seqs, None, false).unwrap();
        for (n, (inp, e)) in inputs.iter().zip(&alone).enumerate() {
            let drift = max_abs(e, &out.embeddings[n * dim..(n + 1) * dim]);
            eprintln!("{}: alone vs batched max abs {drift:.2e}", inp.name);
            if drift > BATCH_ABS {
                failures.push(format!("{}: alone vs batched max abs {drift:.3e}", inp.name));
            }
        }
        finish(failures);
    });
}
