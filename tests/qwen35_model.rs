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
//!   bf16 reference's. `make_reference.py --ids tests/fixtures/qwen35_text_ids.npy --out
//!   target/qwen35_ref_text` with `QWEN35_REF_DIR` pointed at it runs both
//!   tests on 512 natural-text tokens instead of the prompt.
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
use tessl::qwen35::{self, AttnProjLayout, GdnProjLayout};
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
    let (shape, fp32) = load_ref("logits_f32");
    let (_, hf_bf16) = load_ref("logits_bf16");
    let vocab = shape[1];
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let top1_miss = |q: &[f64]| {
        fp32.chunks(vocab)
            .zip(q.chunks(vocab))
            .filter(|(p, q)| argmax(p) != argmax(q))
            .count()
    };
    let kl_hf = kl_rows(&fp32, &hf_bf16, vocab);
    let m_h = top1_miss(&hf_bf16);
    eprintln!(
        "vs fp32 over {} positions: transformers bf16 mean KL {:.3e} (max {:.3e}), top-1 misses {m_h}",
        shape[0],
        mean(&kl_hf),
        kl_hf.iter().cloned().fold(0.0, f64::max),
    );
    let (logits, _) = run(Precision::Bf16);
    let got64: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let kl_tessl = kl_rows(&fp32, &got64, vocab);
    let m_t = top1_miss(&got64);
    eprintln!(
        "vs fp32: tessl bf16 mean KL {:.3e} (max {:.3e}), top-1 misses {m_t}, logits rel err {:.3e}",
        mean(&kl_tessl),
        kl_tessl.iter().cloned().fold(0.0, f64::max),
        rel(&logits, &fp32),
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

/// The real 2B at F32, on the staged and decode entry points the tiny tests
/// cover: `begin` / `advance_to` / `final_norm_f32` on a `load_tower` model
/// give `forward`'s traced final norm, and `prefill` then three `Decode::step`s
/// give `forward`'s last row at every length. Bounds are the tiny tests': the
/// decode test's F32 1e-5, with the same top-1 token wherever the top two
/// logits are further apart than that bound (`check_row`), and 1e-5 for the
/// staged final norm (the same arithmetic as the forward's). Needs only the
/// checkpoint, not the transformers reference. One model is resident at a time:
/// two F32 copies of the 2B would be ~15 GB.
#[test]
#[ignore]
fn f32_staged_and_decode_match_forward_on_the_real_2b() {
    const STEPS: usize = 3;
    const PROMPT: usize = 12;
    let rt = GpuRuntime::new().unwrap();
    let st = checkpoint();
    let cfg = Qwen35Config::qwen35_2b().unwrap();
    let n_layers = cfg.layers.len();
    let hidden = cfg.hidden as usize;
    let vocab = cfg.vocab;
    let ids: Vec<u32> = (0..(PROMPT + STEPS) as u32).map(|i| (i * 7919 + 101) % vocab).collect();

    let want_norm = {
        let model = Qwen35Model::load(&rt, &st, PREFIX, cfg.clone(), Precision::F32).unwrap();
        let (mut dec, first) = model.prefill(&ids[..PROMPT], STEPS as u32).unwrap();
        let mut got = vec![first];
        for k in 0..STEPS {
            got.push(dec.step(ids[PROMPT + k]).unwrap());
        }
        for (k, row) in got.iter().enumerate() {
            let want = forward_last(&model, &ids[..PROMPT + k]);
            check_row(&format!("prefill {PROMPT} + {k} decoded"), row, &want, 1e-5);
        }
        let traced = model.forward(&ids[..PROMPT], true).unwrap();
        traced.trace[n_layers].clone()
    };

    let tower = Qwen35Model::load_tower(&rt, &st, PREFIX, cfg, Precision::F32).unwrap();
    let mut s = tower.begin(&ids[..PROMPT]).unwrap();
    for stop in [1, n_layers / 2, n_layers] {
        s.advance_to(stop).unwrap();
        assert_eq!(s.layers_done(), stop);
    }
    let out = rt.alloc_tensor_f32(&[PROMPT, hidden]).unwrap();
    s.final_norm_f32(&out).unwrap();
    rt.synchronize().unwrap();
    let r = rel32(&out.buffer.read_f32(), &want_norm);
    assert!(r <= 1e-5, "staged final norm vs forward's: rel err {r:.3e} > 1e-5");
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

// ------------------------------------------------------------- hardening ---

/// A conv width the kernel is not compiled for (`conv1d_silu` takes 2..=8)
/// is refused with the config, not part way through a forward.
#[test]
fn conv_widths_the_kernel_lacks_are_refused_at_the_config() {
    let text = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    for kw in [0u32, 1, 9, 64] {
        let bad = text.replace(
            "\"linear_conv_kernel_dim\": 4",
            &format!("\"linear_conv_kernel_dim\": {kw}"),
        );
        assert_ne!(bad, text);
        let m = Qwen35Config::from_config_json(&bad)
            .err()
            .unwrap_or_else(|| panic!("conv_kernel {kw} was accepted"));
        assert!(m.contains("conv_kernel"), "kw {kw}: {m}");
    }
    for kw in [2u32, 8] {
        let ok = text.replace(
            "\"linear_conv_kernel_dim\": 4",
            &format!("\"linear_conv_kernel_dim\": {kw}"),
        );
        Qwen35Config::from_config_json(&ok).unwrap();
    }
    with_gpu(|rt| {
        let cfg = Qwen35Config {
            conv_kernel: 9,
            ..tiny_config()
        };
        assert!(Qwen35Model::random_tower(rt, cfg, Precision::F32, 1).is_err());
    });
}

/// Weights written under a live session or staged prefill (write_parameters,
/// adamw_step) would leave it continuing from state the old weights made:
/// both refuse, before encoding anything, and a fresh prefill works.
#[test]
fn sessions_refuse_to_continue_across_a_weight_write() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = tiny_config();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let ids = seq(6);
        let (mut dec, _) = model.prefill(&ids[..4], 2).unwrap();
        let mut staged = model.begin(&ids).unwrap();
        staged.advance_to(1).unwrap();
        let table = model.parameter_table().unwrap();
        let values: Vec<tessl::Tensor> = table
            .iter()
            .map(|i| rt.alloc_tensor_f32(&i.storage_shape()).unwrap())
            .collect();
        model.read_parameters(&values).unwrap();
        model.write_parameters(&values).unwrap();
        let _ = rt.take_dispatch_count();
        let m = dec.step(ids[4]).unwrap_err();
        assert!(m.contains("parameters were written"), "{m}");
        assert_eq!(dec.position(), 4);
        let m = staged.advance_to(2).unwrap_err();
        assert!(m.contains("parameters were written"), "{m}");
        assert!(staged.logits(LogitRows::Last).is_err());
        assert!(staged.score_answers(&[0], &[1]).is_err());
        let out = rt.alloc_tensor_f32(&[6, cfg.hidden as usize]).unwrap();
        assert!(staged.final_norm_f32(&out).is_err());
        assert_eq!(rt.take_dispatch_count(), 0, "a refused call encoded work");
        let (mut dec, _) = model.prefill(&ids[..4], 2).unwrap();
        let got = dec.step(ids[4]).unwrap();
        assert!(rel32(&got, &forward_last(&model, &ids[..5])) <= 1e-5);
    });
}

/// A prefill whose kept state would not fit the device's recommended working
/// set is refused before any GPU work, as a training step is.
#[test]
fn a_prefill_over_the_working_set_is_refused_before_it_runs() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", tiny_config(), Precision::F32).unwrap();
        let ids = seq(4);
        rt.set_recommended_working_set_for_test(rt.current_allocated_bytes() + (1 << 20));
        let _ = rt.take_dispatch_count();
        // 8M suffix slots of one KV head of 256 f32 K and V: 16 GiB.
        let m = model
            .prefill(&ids, 8 << 20)
            .err()
            .expect("an oversized prefill was accepted");
        assert!(m.contains("working set"), "{m}");
        assert_eq!(rt.take_dispatch_count(), 0, "the refused prefill encoded work");
        model.prefill(&ids, 4).expect("a prefill that fits still runs");
    });
}

/// The bytes a refusal names ("needs N B on top of the H B allocated").
fn refused_need(msg: &str) -> u64 {
    let head = &msg[..msg
        .find(" B on top of")
        .unwrap_or_else(|| panic!("not a working-set refusal: {msg}"))];
    let digits: String = head
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits.parse().unwrap_or_else(|_| panic!("no byte count in {msg}"))
}

/// The pre-flight counts what the prefill allocates, not only what the
/// session keeps: its activations for every token (with the GDN workspace),
/// the decode activations and the logits. A limit that admits a one-token
/// prompt with the same K/V capacity refuses a 4096-token prompt before any
/// GPU work; `forward` and `begin` are refused on their activations alone.
#[test]
fn a_prefill_counts_its_own_activations_against_the_working_set() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", tiny_config(), Precision::F32).unwrap();
        let (long, max_new) = (4096usize, 4u32);
        let ids = seq(long);
        let refuse = |limit_over: u64, f: &dyn Fn() -> Result<(), String>| {
            rt.set_recommended_working_set_for_test(rt.current_allocated_bytes() + limit_over);
            let _ = rt.take_dispatch_count();
            f()
        };
        // What a session of the same K/V capacity needs at one token, from
        // the refusal's own figure.
        let small = refuse(0, &|| model.prefill(&ids[..1], max_new + long as u32 - 1).map(|_| ()))
            .expect_err("a zero limit admitted a prefill");
        let kept = refused_need(&small);
        let margin = 1 << 20;
        let m = refuse(kept + margin, &|| model.prefill(&ids, max_new).map(|_| ()))
            .expect_err("a prefill whose activations do not fit was accepted");
        assert!(m.contains("working set") && m.contains("Qwen35Model::prefill"), "{m}");
        assert!(
            refused_need(&m) > kept + margin,
            "the activations of 4096 tokens were not counted: {m}"
        );
        assert_eq!(rt.take_dispatch_count(), 0, "the refused prefill encoded work");
        // The same limit still admits the small session: the figure is not inflated.
        refuse(kept + margin, &|| {
            model.prefill(&ids[..1], max_new + long as u32 - 1).map(|_| ())
        })
        .expect("a prefill that fits was refused");

        for (what, run) in [
            (
                "forward",
                &(|| model.forward(&ids, false).map(|_| ())) as &dyn Fn() -> Result<(), String>,
            ),
            ("begin", &|| model.begin(&ids).map(|_| ())),
        ] {
            let m = refuse(margin, run).expect_err("an oversized forward was accepted");
            assert!(
                m.contains("working set") && m.contains(&format!("Qwen35Model::{what}")),
                "{m}"
            );
            assert_eq!(rt.take_dispatch_count(), 0, "the refused {what} encoded work");
        }
        rt.set_recommended_working_set_for_test(u64::MAX / 2);
        model.forward(&ids[..8], false).expect("a forward that fits still runs");
    });
}

// ---------------------------------------------------------------- stress ---

/// A tiny deterministic generator (splitmix64) for the randomized shapes.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[(self.next() % from.len() as u64) as usize]
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }
}

/// A random Qwen3.5 shape the kernels are compiled for: 1..=5 layers in any
/// order, grouped GDN and attention heads, every conv width, value dims and
/// rotary widths across their range.
fn random_config(r: &mut Rng) -> Qwen35Config {
    let n_layers = 1 + r.below(5) as usize;
    let layers = (0..n_layers)
        .map(|_| {
            if r.below(2) == 0 {
                LayerKind::LinearAttention
            } else {
                LayerKind::FullAttention
            }
        })
        .collect();
    let k_heads = r.pick(&[1u32, 2]);
    let q_heads = r.pick(&[1u32, 2, 4]);
    let kv_heads = r.pick(&[1u32, 2].map(|d| (q_heads / d).max(1)));
    Qwen35Config {
        hidden: r.pick(&[32u32, 64, 128]),
        intermediate: r.pick(&[64u32, 128, 192]),
        vocab: r.pick(&[17u32, 64, 100]),
        layers,
        gdn: GdnProjLayout::new(k_heads, k_heads * r.pick(&[1u32, 2]), r.pick(&[32u32, 64, 128])).unwrap(),
        conv_kernel: 2 + r.below(7),
        attn: AttnProjLayout::new(q_heads, kv_heads, 256).unwrap(),
        rotary_dim: r.pick(&[2u32, 64, 128, 256]),
        rope_theta: r.pick(&[1e4f32, 1e7]),
        rms_norm_eps: 1e-6,
    }
}

/// `row` matches `want` within `bound`, and has its top-1 token unless the
/// top two of `want` are closer than the bound allows telling apart.
fn check_row(label: &str, row: &[f32], want: &[f32], bound: f64) -> f64 {
    let r = rel32(row, want);
    assert!(r <= bound, "{label}: rel err {r:.3e} > {bound:.0e}");
    let peak = want.iter().fold(0.0f64, |m, &w| m.max(f64::from(w).abs()));
    let mut sorted: Vec<f64> = want.iter().map(|&x| f64::from(x)).collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    if sorted.len() < 2 || sorted[0] - sorted[1] > 2.0 * bound * peak {
        assert_eq!(argmax32(row), argmax32(want), "{label}: top-1");
    }
    r
}

/// Decode against the forward on 40 random F32 shapes, prompts of 1..=150
/// tokens and 1..=6 steps, with the runtime's encoding synchronous and then
/// asynchronous (one command buffer until a wait, so every buffer a call
/// allocates and drops must outlive the work that reads it). Bound: the
/// decode test's F32 1e-5.
#[test]
fn stress_decode_matches_forward_on_random_shapes() {
    let mut r = Rng(0x51ce_d00d);
    // (shape, prompt tokens, steps, weight seed)
    type Case = (Qwen35Config, usize, u32, u64);
    let cases: Vec<Case> = (0..40)
        .map(|_| {
            let cfg = random_config(&mut r);
            let n = 1 + r.below(150) as usize;
            (cfg, n, 1 + r.below(6), r.next())
        })
        .collect();
    // The seed must reach what the sweep claims to cover.
    let covered = |f: &dyn Fn(&Case) -> bool| cases.iter().any(f);
    for kw in qwen35::CONV_KERNEL_WIDTHS {
        assert!(
            covered(&|c| c.0.conv_kernel == kw && c.0.layers.contains(&LayerKind::LinearAttention)),
            "conv {kw}"
        );
    }
    assert!(covered(
        &|c| c.0.gdn.v_heads() > c.0.gdn.k_heads() && c.0.layers.contains(&LayerKind::LinearAttention)
    ));
    assert!(covered(
        &|c| c.0.attn.q_heads() > c.0.attn.kv_heads() && c.0.layers.contains(&LayerKind::FullAttention)
    ));
    assert!(covered(&|c| c.0.layers.len() >= 4 && c.1 > 64));
    assert!(covered(&|c| c.1 < c.0.conv_kernel as usize - 1));
    for async_encode in [false, true] {
        with_gpu(|rt| {
            rt.set_async_encode(async_encode).unwrap();
            let mut worst = 0.0f64;
            for (i, (cfg, n, steps, seed)) in cases.iter().enumerate() {
                let label = format!("case {i} (async {async_encode}, n {n}, steps {steps}, {cfg:?})");
                let model = Qwen35Model::random_tower(rt, cfg.clone(), Precision::F32, *seed).unwrap();
                let v = cfg.vocab;
                let ids: Vec<u32> = (0..*n as u32 + steps).map(|j| (j * 31 + *seed as u32) % v).collect();
                let (mut dec, first) = model.prefill(&ids[..*n], *steps).unwrap();
                worst = worst.max(check_row(&label, &first, &forward_last(&model, &ids[..*n]), 1e-5));
                for k in 0..*steps as usize {
                    let got = dec.step(ids[n + k]).unwrap();
                    let want = forward_last(&model, &ids[..n + k + 1]);
                    worst = worst.max(check_row(&format!("{label} step {k}"), &got, &want, 1e-5));
                }
                assert_eq!(dec.remaining(), 0);
            }
            eprintln!("async {async_encode}: 40 random shapes, worst rel err {worst:.2e}");
            rt.set_async_encode(false).unwrap();
        });
    }
}

/// 140 tokens decoded after a one-token prompt, past two GDN chunk
/// boundaries of the forward it is compared with, at both precisions: a
/// state that drifts, or a suffix slot that wraps, shows up as the steps go.
#[test]
fn stress_long_decode_stays_on_the_forward() {
    const STEPS: u32 = 140;
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        for precision in [Precision::F32, Precision::Bf16] {
            let model = Qwen35Model::load(rt, &st, "model.", tiny_config(), precision).unwrap();
            let ids = seq(1 + STEPS as usize);
            let (mut dec, _) = model.prefill(&ids[..1], STEPS).unwrap();
            let mut worst = 0.0f64;
            for k in 0..STEPS as usize {
                let got = dec.step(ids[1 + k]).unwrap();
                let want = forward_last(&model, &ids[..k + 2]);
                worst = worst.max(check_row(
                    &format!("{precision:?} step {k}"),
                    &got,
                    &want,
                    bound(precision),
                ));
            }
            assert!(dec.step(ids[0]).is_err(), "past max_new");
            eprintln!("{precision:?}: {STEPS} steps, worst rel err {worst:.2e}");
        }
    });
}

/// Three sessions of one model decoded in turn, the forward oracle and a
/// dropped session churning the buffer pool between their steps, under async
/// encode: no session's state may leak into another's, and none may read a
/// buffer the pool handed elsewhere.
#[test]
fn stress_interleaved_sessions_keep_their_own_state() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        rt.set_async_encode(true).unwrap();
        let model = Qwen35Model::load(rt, &st, "model.", tiny_config(), Precision::F32).unwrap();
        let prompts: [Vec<u32>; 3] = [seq(80), seq(3).iter().map(|x| 63 - x).collect(), seq(65)];
        let mut sessions: Vec<_> = prompts
            .iter()
            .map(|p| (p.clone(), model.prefill(p, 12).unwrap().0))
            .collect();
        for k in 0..12u32 {
            for (s, (ids, dec)) in sessions.iter_mut().enumerate() {
                let id = (k * 7 + s as u32 * 13) % 64;
                let got = dec.step(id).unwrap();
                ids.push(id);
                check_row(&format!("session {s} step {k}"), &got, &forward_last(&model, ids), 1e-5);
                // Churn: a session made and dropped, and a forward of another length.
                drop(model.prefill(&seq(1 + (k as usize * 5) % 70), 3).unwrap());
                model.forward(&seq(1 + k as usize), false).unwrap();
            }
        }
        rt.set_async_encode(false).unwrap();
    });
}

/// Switching the runtime to relaxed-f32 GEMMs under an F32 session refuses
/// its steps without breaking it, and switching back continues the sequence.
#[test]
fn stress_relaxed_precision_mid_session_is_refused_not_fatal() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", tiny_config(), Precision::F32).unwrap();
        let ids = seq(10);
        let (mut dec, _) = model.prefill(&ids[..7], 3).unwrap();
        rt.set_relaxed_precision(true);
        let _ = rt.take_dispatch_count();
        let m = dec.step(ids[7]).unwrap_err();
        assert!(m.contains("relaxed precision"), "{m}");
        assert_eq!(rt.take_dispatch_count(), 0);
        rt.set_relaxed_precision(false);
        for k in 7..10 {
            let got = dec.step(ids[k]).unwrap();
            check_row(&format!("step {k}"), &got, &forward_last(&model, &ids[..=k]), 1e-5);
        }
    });
}

/// A step that fails after the first layer leaves the position where it was
/// and poisons the session for good: the next step is refused before any work.
#[test]
fn decode_step_failing_part_way_poisons_the_session() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", tiny_config(), Precision::F32).unwrap();
        let ids = seq(10);
        let (mut dec, _) = model.prefill(&ids[..7], 3).unwrap();
        dec.step(ids[7]).unwrap();
        let at = dec.position();
        model.fail_layer_after_for_test(1);
        let e = dec.step(ids[8]).unwrap_err();
        assert!(e.contains("injected fault"), "{e}");
        assert_eq!(dec.position(), at, "a failed step does not advance");
        let _ = rt.take_dispatch_count();
        let e = dec.step(ids[8]).unwrap_err();
        assert!(e.contains("earlier step failed"), "{e}");
        assert_eq!(rt.take_dispatch_count(), 0, "the refused step encoded work");
        assert_eq!(dec.position(), at);
    });
}

/// A step whose logits cannot be read back after its layers ran does not
/// advance the session: `position()` stays where it was (the token's logits
/// were lost, so counting it decoded would skip it), and the session is
/// poisoned, since its layers' state moved on.
#[test]
fn decode_step_failing_to_read_its_logits_does_not_advance() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", tiny_config(), Precision::F32).unwrap();
        let ids = seq(10);
        let (mut dec, _) = model.prefill(&ids[..7], 3).unwrap();
        dec.step(ids[7]).unwrap();
        let at = (dec.position(), dec.remaining());
        model.fail_logits_read_for_test();
        let e = dec.step(ids[8]).unwrap_err();
        assert!(e.contains("injected fault"), "{e}");
        assert_eq!(
            (dec.position(), dec.remaining()),
            at,
            "a failed read advanced the session"
        );
        let _ = rt.take_dispatch_count();
        let e = dec.step(ids[8]).unwrap_err();
        assert!(e.contains("earlier step failed"), "{e}");
        assert_eq!(rt.take_dispatch_count(), 0, "the refused step encoded work");
    });
}

/// Hostile but valid row and answer selections: the last token id, one row
/// asked for a thousand times, the largest answer set, and one past it.
#[test]
fn stress_hostile_rows_and_answers() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = tiny_config();
    let v = cfg.vocab as usize;
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let ids = vec![cfg.vocab - 1; 9];
        let all = model.forward(&ids, false).unwrap().logits;
        let rows = vec![8u32; 1000];
        let got = model.forward_rows(&ids, LogitRows::Rows(&rows), false).unwrap().logits;
        assert_eq!(got.len(), 1000 * v);
        for chunk in got.chunks(v) {
            assert!(rel32(chunk, &all[8 * v..]) <= 1e-5);
        }
        let mut s = model.begin(&ids).unwrap();
        s.advance_to(cfg.layers.len()).unwrap();
        let max = qwen35::MAX_ANSWERS as usize;
        let answers: Vec<u32> = (0..max as u32).map(|i| i % cfg.vocab).collect();
        let scores = s.score_answers(&[8, 0], &answers).unwrap();
        assert_eq!(scores.logits.len(), 2 * max);
        assert!(scores.logits.iter().chain(&scores.logprobs).all(|x| x.is_finite()));
        for (i, &a) in answers.iter().enumerate().take(v) {
            let w = all[8 * v + a as usize];
            assert!((scores.logits[i] - w).abs() <= 1e-5 * w.abs().max(1.0), "answer {a}");
        }
        let too_many = vec![0u32; max + 1];
        assert!(s.score_answers(&[0], &too_many).is_err());
    });
}

/// Attention heads that do not group (query heads not a multiple of the KV
/// heads) are refused with the config, not by attention part way through a
/// forward.
#[test]
fn ungrouped_attention_heads_are_refused_at_the_config() {
    let text = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    let bad = text
        .replace("\"num_attention_heads\": 2", "\"num_attention_heads\": 3")
        .replace("\"num_key_value_heads\": 1", "\"num_key_value_heads\": 2");
    let m = Qwen35Config::from_config_json(&bad).expect_err("3 query over 2 KV heads was accepted");
    assert!(m.contains("num_key_value_heads") || m.contains("kv_heads"), "{m}");
    with_gpu(|rt| {
        let cfg = Qwen35Config {
            attn: AttnProjLayout::new(3, 2, 256).unwrap(),
            ..tiny_config()
        };
        assert!(Qwen35Model::random_tower(rt, cfg, Precision::F32, 1).is_err());
    });
}

/// What `forward_rows`, `prefill` and `score_answers` can tell is wrong from
/// their arguments is refused before the forward runs: no dispatch at all.
#[test]
fn argument_refusals_encode_nothing() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = tiny_config();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let ids = seq(5);
        let _ = rt.take_dispatch_count();
        assert!(model.forward_rows(&ids, LogitRows::Rows(&[1, 5]), false).is_err());
        assert_eq!(rt.take_dispatch_count(), 0, "forward_rows ran before refusing a row");
        let mut s = model.begin(&ids).unwrap();
        s.advance_to(cfg.layers.len()).unwrap();
        let before = rt.current_allocated_bytes();
        let huge = vec![0u32; qwen35::MAX_ANSWERS as usize + 1];
        let m = s
            .score_answers(&vec![0u32; 1 << 16], &huge)
            .err()
            .expect("4097 answers were accepted");
        assert!(m.contains("answers"), "{m}");
        assert!(
            rt.current_allocated_bytes() <= before,
            "score_answers allocated before refusing"
        );
    });
}

// ------------------------------------------ unread tensors and config keys ---

/// `st`'s tensors plus `extra` `(name, shape)` F32 zeros, as a checkpoint.
fn checkpoint_with(st: &SafeTensors, extra: &[(&str, &[usize])]) -> SafeTensors {
    let mut entries: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
    for name in st.names() {
        let (info, bytes) = st.read_raw(name).unwrap();
        let dtype = format!("{:?}", info.dtype).to_uppercase();
        assert!(dtype == "F32" || dtype == "BF16", "{name}: fixture dtype {dtype}");
        entries.push((name.to_string(), dtype, info.shape, bytes));
    }
    for &(name, shape) in extra {
        let n: usize = shape.iter().product();
        entries.push((name.to_string(), "F32".into(), shape.to_vec(), vec![0u8; n * 4]));
    }
    let (mut header, mut data) = (String::from("{"), Vec::new());
    for (i, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
        let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
        header += &format!(
            "{}\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            if i > 0 { "," } else { "" },
            dims.join(","),
            data.len(),
            data.len() + bytes.len()
        );
        data.extend_from_slice(bytes);
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    SafeTensors::from_bytes("qwen35 loader test", out).unwrap()
}

/// A tensor under the prefix the model never reads is refused by name, for
/// each way it can arise: a layer past the config's stack (a config shorter
/// than the checkpoint), a projection bias, an extra norm. Tensors outside
/// the prefix are not the loader's business.
#[test]
fn tensors_the_model_never_reads_are_refused_by_name() {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let full = tiny_config();
    let shorter = Qwen35Config {
        layers: vec![LayerKind::LinearAttention],
        ..tiny_config()
    };
    with_gpu(|rt| {
        // The unmodified checkpoint loads.
        Qwen35Model::load(rt, &st, "model.", full.clone(), Precision::F32).unwrap();

        // (a) Fewer layers than the checkpoint: layer 1 is dropped unless refused.
        let m = Qwen35Model::load(rt, &st, "model.", shorter, Precision::F32)
            .err()
            .expect("a config with fewer layers than the checkpoint loaded");
        assert!(m.contains("model.layers.1."), "{m}");

        // (b) A stray bias, an extra norm, an extra layer's tensor.
        for stray in [
            "model.layers.1.self_attn.q_proj.bias",
            "model.layers.0.linear_attn.extra_norm.weight",
            "model.layers.2.mlp.up_proj.weight",
        ] {
            let bad = checkpoint_with(&st, &[(stray, &[4])]);
            for precision in [Precision::F32, Precision::Bf16] {
                let m = Qwen35Model::load(rt, &bad, "model.", full.clone(), precision)
                    .err()
                    .unwrap_or_else(|| panic!("{stray}: loaded at {precision:?}"));
                assert!(m.contains(stray), "{stray}: error does not name it: {m}");
            }
            let m = Qwen35Model::load_tower(rt, &bad, "model.", full.clone(), Precision::F32)
                .err()
                .unwrap_or_else(|| panic!("{stray}: load_tower accepted it"));
            assert!(m.contains(stray), "{stray}: load_tower error does not name it: {m}");
        }

        // Outside the prefix: the visual tower and `mtp.` ride along.
        let ok = checkpoint_with(&st, &[("visual.blocks.0.attn.qkv.bias", &[4]), ("mtp.fc.weight", &[4])]);
        Qwen35Model::load(rt, &ok, "model.", full, Precision::F32).unwrap();
    });
}

fn with_key(text: &str, key_line: &str) -> String {
    let anchor = "\"head_dim\": 256,";
    assert!(text.contains(anchor));
    text.replacen(anchor, &format!("{anchor}\n {key_line},"), 1)
}

/// `mamba_ssm_dtype` is accepted absent or `"float32"` (transformers casts
/// the GDN rules to float32 unconditionally) and refused, by name, otherwise.
#[test]
fn mamba_ssm_dtype_other_than_float32_is_refused_at_the_config() {
    let text = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    assert!(!text.contains("mamba_ssm_dtype"));
    Qwen35Config::from_config_json(&text).expect("an absent mamba_ssm_dtype is float32");
    Qwen35Config::from_config_json(&with_key(&text, "\"mamba_ssm_dtype\": \"float32\"")).unwrap();
    for bad in ["\"bfloat16\"", "\"float16\"", "\"\"", "1", "null"] {
        let m = Qwen35Config::from_config_json(&with_key(&text, &format!("\"mamba_ssm_dtype\": {bad}")))
            .err()
            .unwrap_or_else(|| panic!("mamba_ssm_dtype {bad} was accepted"));
        assert!(m.contains("mamba_ssm_dtype"), "{bad}: {m}");
    }
}

/// `attention_dropout` is accepted only as 0 (the fixture carries 0.0).
#[test]
fn nonzero_attention_dropout_is_refused_at_the_config() {
    let text = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    assert!(text.contains("\"attention_dropout\": 0.0,"));
    for bad in ["0.1", "1e-9", "1", "-0.5", "\"0.0\"", "null"] {
        let t = text.replace("\"attention_dropout\": 0.0,", &format!("\"attention_dropout\": {bad},"));
        let m = Qwen35Config::from_config_json(&t)
            .err()
            .unwrap_or_else(|| panic!("attention_dropout {bad} was accepted"));
        assert!(m.contains("attention_dropout"), "{bad}: {m}");
    }
    let t = text.replace("\"attention_dropout\": 0.0,", "");
    Qwen35Config::from_config_json(&t).expect("an absent attention_dropout is 0");
}

/// The real 2B's `config.json` (text_config with `mamba_ssm_dtype: float32`
/// and `attention_dropout: 0.0`) still loads.
#[test]
fn the_real_2b_config_shape_still_loads() {
    let text = r#"{"model_type":"qwen3_5","tie_word_embeddings":true,"text_config":{
        "attention_bias":false,"attention_dropout":0.0,"attn_output_gate":true,"dtype":"bfloat16",
        "full_attention_interval":4,"head_dim":256,"hidden_act":"silu","hidden_size":2048,
        "intermediate_size":6144,"linear_conv_kernel_dim":4,"linear_key_head_dim":128,
        "linear_num_key_heads":16,"linear_num_value_heads":16,"linear_value_head_dim":128,
        "mlp_only_layers":[],"mtp_num_hidden_layers":1,"num_attention_heads":8,"num_hidden_layers":24,
        "num_key_value_heads":2,"rms_norm_eps":1e-06,"tie_word_embeddings":true,"vocab_size":248320,
        "mamba_ssm_dtype":"float32","rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],
        "rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}"#;
    let cfg = Qwen35Config::from_config_json(text).unwrap();
    assert_eq!(cfg.layers.len(), 24);
}
