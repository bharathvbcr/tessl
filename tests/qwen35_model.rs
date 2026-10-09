//! The Qwen3.5 text model (`tessl::qwen35_model`) end to end.
//!
//! **Against transformers** (opt-in): the whole Qwen3.5-2B forward. Every
//! kernel is checked against transformers on its own; this checks their
//! composition: layer order, norm offsets, weight layouts, the tied LM head.
//! It needs the real checkpoint and the reference outputs
//! `tools/qwen35_ref/make_reference.py` writes, so it is ignored by default:
//!
//! ```text
//! python3 tools/qwen35_ref/make_reference.py      # once: target/qwen35_ref/
//! QWEN35_2B_SAFETENSORS=/path/to/model.safetensors-00001-of-00001.safetensors \
//!   cargo test --release --test qwen35_model -- --ignored --test-threads=1
//! ```
//!
//! `QWEN35_REF_DIR` overrides `target/qwen35_ref`. A missing variable or file
//! fails the test rather than skipping it.
//!
//! Bounds, fixed before any run:
//!
//! * `F32`: the same arithmetic as transformers' fp32 forward up to operation
//!   order. Each layer's residual stream within 1e-4 of its largest magnitude,
//!   the logits within 1e-3 of the largest logit, and the same top-1 token at
//!   every position. (The kernels agree with f64 to ~1e-7 relative each; 24
//!   layers of reordering leave orders of magnitude of margin.)
//! * `Bf16`: bf16 GEMM inputs are a real change of numerics, so it is held to
//!   transformers' own bf16 forward instead: its mean KL divergence from the
//!   fp32 reference and its top-1 disagreements with it may not exceed the
//!   bf16 reference's.
//!
//! **Against the forward itself** (always run, on the tiny two-layer
//! checkpoint in `tests/fixtures/qwen35_train/` and a seeded four-layer
//! tower with grouped GDN heads): decode, chosen logit rows and answer
//! scoring each give what `forward` gives. Bounds, fixed before any run, as
//! `max |got - want| / max |want|` over the compared logits:
//!
//! * Decode (`prefill(N)` then steps, against `forward(N + k)`'s last row):
//!   `F32` 1e-5 and the same top-1 token. Decode runs the recurrent GDN rule
//!   and split-KV attention where the prefill runs the chunked rule and tiled
//!   attention, which agree with f64 to ~1e-7 relative each, so a few layers
//!   of different summation order stay two orders of magnitude inside it,
//!   while a wrong state, position or slot moves a logit by its own size.
//!   `Bf16` 2e-2: the two paths' f32 values can round to neighbouring bf16
//!   values (2^-9 relative) on the way into a GEMM.
//! * Chosen rows (`forward_rows`, `Staged::logits`) against the same rows of
//!   `forward`: 1e-5 at both precisions; the rows' arithmetic is the same,
//!   and only the head GEMM's row count differs.
//! * Answer scores against `forward`'s logits at the answer tokens: `F32`
//!   1e-5; `Bf16` 2e-2, since the scoring kernel keeps the normed row f32
//!   where the bf16 head GEMM rounds it to bf16 (2^-9 relative per element).
//!   The log-probabilities are held to the log-softmax of `forward`'s answer
//!   logits within the same bound, in absolute terms.

mod common;

use std::path::{Path, PathBuf};

use common::with_gpu;
use tessl::npy::read_npy;
use tessl::qwen35::{AttnProjLayout, GdnProjLayout};
use tessl::qwen35_model::{LayerKind, LogitRows, Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

const PREFIX: &str = "model.language_model.";

fn ref_dir() -> PathBuf {
    std::env::var_os("QWEN35_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/qwen35_ref"))
}

fn load_ref(name: &str) -> (Vec<usize>, Vec<f64>) {
    let path = ref_dir().join(format!("{name}.npy"));
    let a = read_npy(&path).unwrap_or_else(|e| panic!("{e}\n(run python3 tools/qwen35_ref/make_reference.py first)"));
    let data = if let Ok(s) = a.f32_slice() {
        s.iter().map(|&x| x as f64).collect()
    } else if let Some(i) = &a.data_i64 {
        i.iter().map(|&x| x as f64).collect()
    } else {
        panic!("{name}: unexpected dtype")
    };
    (a.shape, data)
}

fn checkpoint() -> SafeTensors {
    let path = std::env::var("QWEN35_2B_SAFETENSORS")
        .expect("set QWEN35_2B_SAFETENSORS to the Qwen3.5-2B-Base .safetensors file");
    SafeTensors::open(Path::new(&path)).unwrap()
}

fn ids() -> Vec<u32> {
    let (_, ids) = load_ref("ids");
    ids.iter()
        .map(|&x| u32::try_from(x as i64).expect("token id fits u32"))
        .collect()
}

/// max |got - want| / max |want|.
fn rel(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, &w| m.max(w.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((g as f64 - w).abs()));
    err / scale.max(f64::MIN_POSITIVE)
}

fn argmax(row: &[f64]) -> usize {
    row.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0
}

fn log_softmax(row: &[f64]) -> Vec<f64> {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    row.iter().map(|v| v - lse).collect()
}

/// KL(p || q) from logits, per row.
fn kl_rows(p_logits: &[f64], q_logits: &[f64], vocab: usize) -> Vec<f64> {
    p_logits
        .chunks(vocab)
        .zip(q_logits.chunks(vocab))
        .map(|(p, q)| {
            let (lp, lq) = (log_softmax(p), log_softmax(q));
            lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
        })
        .collect()
}

fn run(precision: Precision) -> (Vec<f32>, Vec<Vec<f32>>) {
    let rt = GpuRuntime::new().unwrap();
    let st = checkpoint();
    let model = Qwen35Model::load(&rt, &st, PREFIX, Qwen35Config::qwen35_2b().unwrap(), precision).unwrap();
    let out = model.forward(&ids(), true).unwrap();
    assert!(out.logits.iter().all(|x| x.is_finite()), "non-finite logits");
    (out.logits, out.trace)
}

#[test]
#[ignore]
fn f32_forward_matches_transformers_fp32_layer_by_layer() {
    let (logits, trace) = run(Precision::F32);
    let cfg = Qwen35Config::qwen35_2b().unwrap();
    let n_layers = cfg.layers.len();
    assert_eq!(trace.len(), n_layers + 1, "trace: every layer, then the final norm");
    let mut worst = 0.0f64;
    for (l, got) in trace[..n_layers - 1].iter().enumerate() {
        let (_, want) = load_ref(&format!("hidden_f32_{l}"));
        let r = rel(got, &want);
        eprintln!("layer {l:2} ({:?}): rel err {r:.2e}", cfg.layers[l]);
        worst = worst.max(r);
        assert!(r <= 1e-4, "layer {l}: residual stream rel err {r:.3e} > 1e-4");
    }
    // transformers reports the final norm's output in place of the last
    // layer's residual stream.
    let (_, want) = load_ref("final_norm_f32");
    let r = rel(&trace[n_layers], &want);
    eprintln!("final norm: rel err {r:.2e} (worst layer {worst:.2e})");
    assert!(r <= 1e-4, "final norm rel err {r:.3e} > 1e-4");

    let (shape, want) = load_ref("logits_f32");
    let vocab = shape[1];
    let r = rel(&logits, &want);
    let got64: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let mismatched: Vec<usize> = got64
        .chunks(vocab)
        .zip(want.chunks(vocab))
        .enumerate()
        .filter(|(_, (g, w))| argmax(g) != argmax(w))
        .map(|(t, _)| t)
        .collect();
    let kl = kl_rows(&want, &got64, vocab);
    let kl_max = kl.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "logits: rel err {r:.2e}, top-1 mismatches {}/{}, max KL {kl_max:.2e}",
        mismatched.len(),
        shape[0]
    );
    assert!(r <= 1e-3, "logits rel err {r:.3e} > 1e-3");
    assert!(mismatched.is_empty(), "top-1 differs at positions {mismatched:?}");
}

#[test]
#[ignore]
fn bf16_forward_is_no_worse_than_transformers_bf16() {
    let (logits, _) = run(Precision::Bf16);
    let (shape, fp32) = load_ref("logits_f32");
    let (_, hf_bf16) = load_ref("logits_bf16");
    let vocab = shape[1];
    let got64: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let kl_tessl = kl_rows(&fp32, &got64, vocab);
    let kl_hf = kl_rows(&fp32, &hf_bf16, vocab);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let top1_miss = |q: &[f64]| {
        fp32.chunks(vocab)
            .zip(q.chunks(vocab))
            .filter(|(p, q)| argmax(p) != argmax(q))
            .count()
    };
    let (m_t, m_h) = (top1_miss(&got64), top1_miss(&hf_bf16));
    eprintln!(
        "vs fp32 over {} positions: tessl bf16 mean KL {:.3e} (max {:.3e}), top-1 misses {m_t}; \
         transformers bf16 mean KL {:.3e} (max {:.3e}), top-1 misses {m_h}",
        shape[0],
        mean(&kl_tessl),
        kl_tessl.iter().cloned().fold(0.0, f64::max),
        mean(&kl_hf),
        kl_hf.iter().cloned().fold(0.0, f64::max),
    );
    assert!(
        mean(&kl_tessl) <= mean(&kl_hf),
        "tessl bf16 is further from fp32 than transformers' bf16 forward"
    );
    assert!(
        m_t <= m_h,
        "tessl bf16 misses more top-1 tokens ({m_t}) than transformers bf16 ({m_h})"
    );
}

// ------------------------------------------------- decode, rows and answers ---

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn tiny_config() -> Qwen35Config {
    Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap()
}

/// Four layers (GDN, attention, GDN, attention) at the tiny model's widths,
/// with two GDN value heads per key head (the 4B's grouping) where the
/// fixture has one: the decode path's head repetition, and a layer order
/// longer than one of each.
fn grouped_config() -> Qwen35Config {
    let cfg = Qwen35Config {
        layers: vec![
            LayerKind::LinearAttention,
            LayerKind::FullAttention,
            LayerKind::LinearAttention,
            LayerKind::FullAttention,
        ],
        gdn: GdnProjLayout::new(1, 2, 128).unwrap(),
        attn: AttnProjLayout::new(2, 1, 256).unwrap(),
        ..tiny_config()
    };
    assert_eq!(cfg.vocab, 64);
    cfg
}

/// Deterministic ids below the tiny vocabulary of 64.
fn seq(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 64).collect()
}

/// max |got - want| / max |want|, in f64.
fn rel32(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0.0f64, |m, &w| m.max(f64::from(w).abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((f64::from(g) - f64::from(w)).abs()));
    err / peak.max(f64::MIN_POSITIVE)
}

fn argmax32(row: &[f32]) -> usize {
    let wide: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
    argmax(&wide)
}

/// The models the always-run tests cover: the fixture at both precisions
/// (with its packed bf16 head), and the grouped tower at F32 (whose head is
/// the embedding itself).
fn models(rt: &std::sync::Arc<GpuRuntime>) -> Vec<(&'static str, Qwen35Model)> {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    vec![
        (
            "fixture f32",
            Qwen35Model::load(rt, &st, "model.", tiny_config(), Precision::F32).unwrap(),
        ),
        (
            "fixture bf16",
            Qwen35Model::load(rt, &st, "model.", tiny_config(), Precision::Bf16).unwrap(),
        ),
        (
            "grouped f32",
            Qwen35Model::random_tower(rt, grouped_config(), Precision::F32, 7).unwrap(),
        ),
    ]
}

fn bound(precision: Precision) -> f64 {
    match precision {
        Precision::F32 => 1e-5,
        Precision::Bf16 => 2e-2,
    }
}

/// The last row of `forward(ids)`'s logits.
fn forward_last(model: &Qwen35Model, ids: &[u32]) -> Vec<f32> {
    let v = model.config().vocab as usize;
    let all = model.forward(ids, false).unwrap().logits;
    all[(ids.len() - 1) * v..].to_vec()
}

/// `prefill(N)` and then three decode steps give `forward(N + k)`'s last
/// row at every k, from N = 1 (a conv history shorter than its kernel) past
/// the GDN chunk boundary (64). Three steps, because the conv state
/// alternates between two buffers and the attention suffix grows a slot per
/// step: one step cannot see a fault in either.
#[test]
fn prefill_then_decode_matches_forward_at_the_last_row() {
    const STEPS: u32 = 3;
    with_gpu(|rt| {
        for (name, model) in models(rt) {
            let b = bound(model.precision());
            let mut worst = 0.0f64;
            for n in [1usize, 2, 3, 5, 63, 64, 65, 70] {
                let ids = seq(n + STEPS as usize);
                let (mut dec, first) = model.prefill(&ids[..n], STEPS).unwrap();
                assert_eq!((dec.position(), dec.remaining()), (n as u32, STEPS));
                let mut got = vec![first];
                for k in 0..STEPS as usize {
                    got.push(dec.step(ids[n + k]).unwrap());
                }
                assert_eq!((dec.position(), dec.remaining()), (n as u32 + STEPS, 0));
                for (k, row) in got.iter().enumerate() {
                    let want = forward_last(&model, &ids[..n + k]);
                    let r = rel32(row, &want);
                    worst = worst.max(r);
                    assert!(r <= b, "{name}: prefill {n} + {k} decoded: rel err {r:.3e} > {b:.0e}");
                    if model.precision() == Precision::F32 {
                        assert_eq!(argmax32(row), argmax32(&want), "{name}: prefill {n} + {k}: top-1");
                    }
                }
            }
            eprintln!("{name}: decode vs forward, worst rel err {worst:.2e} (bound {b:.0e})");
        }
    });
}

/// `forward_rows` and `Staged::logits` return `forward`'s rows for the
/// positions chosen, in the order chosen, repeats included.
#[test]
fn chosen_logit_rows_are_forwards_rows() {
    with_gpu(|rt| {
        for (name, model) in models(rt) {
            let v = model.config().vocab as usize;
            for t in [1usize, 5, 70] {
                let ids = seq(t);
                let all = model.forward(&ids, true).unwrap();
                let row = |p: usize| &all.logits[p * v..(p + 1) * v];
                let picks: Vec<u32> = [t - 1, 0, t / 2, t - 1].iter().map(|&p| p as u32).collect();
                for (label, rows, want) in [
                    ("all", LogitRows::All, all.logits.clone()),
                    ("last", LogitRows::Last, row(t - 1).to_vec()),
                    (
                        "chosen",
                        LogitRows::Rows(&picks),
                        picks.iter().flat_map(|&p| row(p as usize).to_vec()).collect(),
                    ),
                ] {
                    let out = model.forward_rows(&ids, rows, true).unwrap();
                    assert_eq!(out.logits.len(), want.len(), "{name} t={t} {label}: row count");
                    let r = rel32(&out.logits, &want);
                    assert!(r <= 1e-5, "{name} t={t} {label}: rel err {r:.3e}");
                    // The trace does not depend on the rows chosen.
                    assert_eq!(out.trace.len(), all.trace.len());
                    for (a, b) in out.trace.iter().zip(&all.trace) {
                        assert!(
                            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "{name} {label}: trace"
                        );
                    }
                    let mut s = model.begin(&ids).unwrap();
                    s.advance_to(model.config().layers.len()).unwrap();
                    let staged = s.logits(rows).unwrap();
                    assert_eq!(
                        staged.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        out.logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "{name} t={t} {label}: Staged::logits vs forward_rows"
                    );
                }
                let none = model.forward_rows(&ids, LogitRows::Rows(&[]), false).unwrap();
                assert!(none.logits.is_empty());
            }
        }
    });
}

/// `Staged::score_answers` gives `forward`'s logits at the answer tokens, and
/// their log-softmax over the answers alone, on a full model and on a bf16
/// tower that has no head to run `forward` with.
#[test]
fn answer_scores_are_forwards_logits_at_the_answers() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = tiny_config();
    let n_layers = cfg.layers.len();
    let v = cfg.vocab as usize;
    let answers = [3u32, 17, 42, 63, 3];
    with_gpu(|rt| {
        for precision in [Precision::F32, Precision::Bf16] {
            let full = Qwen35Model::load(rt, &st, "model.", cfg.clone(), precision).unwrap();
            let tower = Qwen35Model::load_tower(rt, &st, "model.", cfg.clone(), precision).unwrap();
            let b = bound(precision);
            for t in [1usize, 9, 70] {
                let ids = seq(t);
                let logits = full.forward(&ids, false).unwrap().logits;
                let rows: Vec<u32> = [t - 1, 0, t - 1].iter().map(|&p| p as u32).collect();
                for model in [&full, &tower] {
                    let mut s = model.begin(&ids).unwrap();
                    s.advance_to(n_layers).unwrap();
                    let got = s.score_answers(&rows, &answers).unwrap();
                    assert_eq!(got.logits.len(), rows.len() * answers.len());
                    for (i, &p) in rows.iter().enumerate() {
                        let full_row = &logits[p as usize * v..(p as usize + 1) * v];
                        let want: Vec<f32> = answers.iter().map(|&a| full_row[a as usize]).collect();
                        let peak = full_row.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
                        let row = &got.logits[i * answers.len()..(i + 1) * answers.len()];
                        let err = row
                            .iter()
                            .zip(&want)
                            .fold(0.0f64, |m, (&g, &w)| m.max((f64::from(g) - f64::from(w)).abs()));
                        assert!(
                            err <= b * peak,
                            "{precision:?} t={t} row {p}: {err:.3e} vs peak {peak:.3e}"
                        );
                        let wide: Vec<f64> = want.iter().map(|&x| f64::from(x)).collect();
                        let want_lp = log_softmax(&wide);
                        let lp = &got.logprobs[i * answers.len()..(i + 1) * answers.len()];
                        for (g, w) in lp.iter().zip(&want_lp) {
                            assert!(
                                (f64::from(*g) - w).abs() <= b,
                                "{precision:?} t={t} row {p}: logprob {g} vs {w}"
                            );
                        }
                    }
                }
            }
        }
    });
}

/// Everything refused is refused before it changes anything: a session keeps
/// decoding after a refused step, and stays at its position.
#[test]
fn decode_and_row_refusals() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = tiny_config();
    let n_layers = cfg.layers.len();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let ids = seq(6);
        assert!(model.prefill(&ids, 0).is_err(), "max_new 0");
        assert!(model.prefill(&[], 2).is_err(), "no tokens");
        assert!(model.prefill(&[cfg.vocab], 2).is_err(), "id past vocab");
        let tower = Qwen35Model::load_tower(rt, &st, "model.", cfg.clone(), Precision::Bf16).unwrap();
        let m = tower.prefill(&ids, 2).err().expect("a bf16 tower has no head");
        assert!(m.contains("load_tower"), "{m}");
        assert!(tower.forward_rows(&ids, LogitRows::Last, false).is_err());

        let (mut dec, _) = model.prefill(&ids[..4], 2).unwrap();
        assert!(dec.step(cfg.vocab).is_err(), "id past vocab");
        assert_eq!(dec.position(), 4, "a refused step must not move");
        // Still continues the sequence after the refusal.
        let got = dec.step(ids[4]).unwrap();
        assert!(rel32(&got, &forward_last(&model, &ids[..5])) <= 1e-5);
        dec.step(ids[5]).unwrap();
        let m = dec.step(ids[0]).unwrap_err();
        assert!(m.contains("max_new"), "{m}");
        assert_eq!((dec.position(), dec.remaining()), (6, 0));

        let mut s = model.begin(&ids).unwrap();
        s.advance_to(n_layers).unwrap();
        let m = s.logits(LogitRows::Rows(&[0, 6])).unwrap_err();
        assert!(m.contains("position 6 >= 6"), "{m}");
        assert!(model.forward_rows(&ids, LogitRows::Rows(&[6]), false).is_err());
        assert!(s.score_answers(&[6], &[1]).is_err(), "position past the sequence");
        assert!(s.score_answers(&[0], &[cfg.vocab]).is_err(), "answer past vocab");
        assert!(s.score_answers(&[0], &[]).is_err(), "no answers");
        assert!(s.score_answers(&[], &[1]).unwrap().logits.is_empty());
    });
}
