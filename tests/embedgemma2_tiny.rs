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
//!
//! `Bf16` (`GemmOperands::Bf16`: the projections stay bf16 on the device and
//! every GEMM input is rounded to bf16; accumulation, the residual stream, the
//! norms and attention stay f32) runs the same checks against the same f64
//! forward, with bounds derived from the rounding rather than tuned, fixed
//! before its first run. The projections here are stored as bf16, as the
//! checkpoint's are, so the f64 forward uses exactly the weights the device
//! holds and only the activations round:
//!
//! * a bf16 rounding is a relative error of at most `u = 2^-8` (3.9e-3) and
//!   about `u / sqrt(3)` (2.3e-3) rms for values spread across a binade. A
//!   GEMM output sums K terms each carrying such an error, which add as a
//!   random walk, as the terms themselves do, so the output's error is about
//!   2.3e-3 of its own rms whatever K is;
//! * an attention branch rounds its input, the attention output and meets
//!   the softmax, about four sources: 5e-3 of the branch rms, and the branch is
//!   rms-normed afterwards (scale-free, O(1) per element). A maximum over some
//!   thousand elements sits at about 4 rms, 2e-2 of the branch;
//! * 3 layers of 3 branches add in quadrature to about 3x one branch (6e-2 of
//!   a branch's scale), and the residual's largest element is about 3x its
//!   rms: about 2e-2 of that maximum. The bound is `5e-2` of it, 2.5x of margin
//!   and still far below a wrong layer, slice or orientation (O(1));
//! * an embedding is a mean over tokens, projected and normalized: about 5e-3
//!   relative rms, `1 - cos ~ theta^2 / 2 ~ 1e-5`. The bound is cosine
//!   `>= 0.9999` (10x of margin; the floor the real model is held to) and max
//!   abs `1e-2` on the unit vector (cosine 0.9999 is an L2 distance of 0.014,
//!   and no element can be further than that);
//! * alone vs batched stays `1e-5`: the bf16 GEMM sums K in the same order
//!   whatever M is, so the f32 values that round are the same bits.

mod common;

use common::embedgemma2::{
    attn_ref, gelu_tanh, l2_normalize, lin, random_weights, rms, rope, safetensors_bytes, Attn, Weights, CONFIG, PREFIX,
};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, DEFAULT_MAX_TOKENS, MAX_BATCH_TOKENS};
use tessl::gemm::GemmOperands;
use tessl::infer_trace;
use tessl::safetensors::SafeTensors;

const BATCH_ABS: f64 = 1e-5;
const MODES: [GemmOperands; 2] = [GemmOperands::ExactF32, GemmOperands::Bf16];

/// What a mode is held to against the f64 forward.
#[derive(Clone, Copy)]
struct Bounds {
    /// Each layer's residual and the final norm, of their largest magnitude.
    layer_rel: f64,
    /// Embeddings, max abs.
    emb_abs: f64,
    /// Embeddings, cosine at least.
    emb_cos: f64,
}

fn bounds(operands: GemmOperands) -> Bounds {
    match operands {
        GemmOperands::ExactF32 => Bounds {
            layer_rel: 1e-4,
            emb_abs: 1e-4,
            emb_cos: 0.99999,
        },
        GemmOperands::Bf16 => Bounds {
            layer_rel: 5e-2,
            emb_abs: 1e-2,
            emb_cos: 0.9999,
        },
    }
}

/// The tiny checkpoint's weights for a mode: the `Bf16` load reads bf16
/// projections only.
fn weights_for(cfg: &EmbedGemma2Config, seed: u64, operands: GemmOperands) -> Weights {
    random_weights(cfg, seed, operands == GemmOperands::Bf16)
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
    matches_the_f64_forward(GemmOperands::ExactF32);
}

/// The same checks at bf16 operands, against the f64 forward over the same
/// bf16 weights, under the derived bounds in the module docs.
#[test]
fn tiny_checkpoint_bf16_matches_the_f64_forward() {
    matches_the_f64_forward(GemmOperands::Bf16);
}

fn matches_the_f64_forward(operands: GemmOperands) {
    let Bounds {
        layer_rel,
        emb_abs,
        emb_cos,
    } = bounds(operands);
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let widths: Vec<u32> = cfg.layers.iter().map(|s| s.kv_heads * s.head_dim).collect();
    assert_eq!(
        widths,
        [512, 256, 512],
        "test bug: the config must mix K/V widths, narrow after wide"
    );
    let w = weights_for(&cfg, 2026, operands);
    let st = open("ok", safetensors_bytes(&w, &[("vision_tower.unused.weight", &[3])]));
    let seqs = sequences(cfg.vocab);
    let (h, dim) = (cfg.hidden as usize, cfg.embedding_dim as usize);

    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
        assert_eq!(model.operands(), operands);

        // 1. Layer by layer, the longest sequence (past the window both ways).
        let (want_trace, _) = host_forward(&cfg, &w, &seqs[0]);
        let out = model.encode(&[&seqs[0]], None, true).unwrap();
        assert_eq!(out.trace.len(), cfg.layers.len() + 1);
        for (i, (got, want)) in out.trace.iter().zip(&want_trace).enumerate() {
            let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let err = max_abs(&got[..seqs[0].len() * h], want);
            eprintln!("trace {i}: max abs {err:.3e} of scale {scale:.3e}");
            assert!(
                err <= layer_rel * scale,
                "trace {i}: max abs {err:.3e} > {layer_rel:.0e} * {scale:.3e}"
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
                err <= emb_abs && cos >= emb_cos,
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
                err <= emb_abs && cos >= emb_cos,
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
                    err <= emb_abs && cos >= emb_cos,
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
/// the host (each forward adds only its four small input and output
/// buffers), and poisoning every unzeroed allocation with NaN changes no bit
/// of any embedding: no kernel reads an element its forward did not write.
/// Before the activations were reused, each forward allocated its own (the
/// two-forward encode made twice the one-forward encode's allocations) and
/// zeroed them on the host.
fn encode_reuses_unzeroed_activations_at(operands: GemmOperands) {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = weights_for(&cfg, 7, operands);
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
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
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
        // Each forward past the first adds only its own small host-visible
        // buffers (ids, lens, segments, embeddings: `ForwardIo`), not another
        // set of activations.
        let io_buffers = 4 * u64::from(two.forwards - 1);
        assert_eq!(
            t2.cold_allocs,
            t1.cold_allocs + io_buffers,
            "two forwards made {} allocations, one made {}: more than {io_buffers} extra, so the \
             activations are not shared",
            t2.cold_allocs,
            t1.cold_allocs
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

/// An `encode` that runs several forwards queues them all and waits for the
/// GPU once, not once per forward, and its embeddings are bit for bit those
/// the same encode gives with the runtime serialized (every dispatch commits
/// and waits, so nothing can overlap), with every unzeroed activation
/// poisoned. It does so on a default runtime and with `async_encode` on. Each
/// sequence also still matches itself run alone. Before, every forward ended
/// in its own `synchronize` and readback, so an encode of N forwards waited N
/// times.
fn multi_forward_encode_waits_once_and_matches_synchronous_at(operands: GemmOperands) {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = weights_for(&cfg, 11, operands);
    let st = open("async_forwards", safetensors_bytes(&w, &[]));
    // Three padded shapes: a long sequence alone, mid-length ones, short ones.
    let seq = |n: u32, s: u32| -> Vec<u32> { (0..n).map(|i| (i * 5 + s * 3 + 1) % cfg.vocab).collect() };
    let long = seq(4000, 0);
    let mid: Vec<Vec<u32>> = (0..3).map(|s| seq(300, s)).collect();
    let short: Vec<Vec<u32>> = (0..7).map(|s| seq(30, s + 5)).collect();
    let mut batch: Vec<&[u32]> = vec![&long];
    batch.extend(mid.iter().map(Vec::as_slice));
    batch.extend(short.iter().map(Vec::as_slice));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
        rt.set_serialized_dispatch(true).unwrap();
        let (sync, t_sync) = traced(|| model.encode(&batch, None, false).unwrap());
        rt.set_serialized_dispatch(false).unwrap();
        assert!(
            sync.forwards >= 3,
            "test bug: {} forwards, need at least 3",
            sync.forwards
        );
        assert!(
            t_sync.sync_waits >= u64::from(sync.forwards),
            "test bug: the serialized encode waited only {} times",
            t_sync.sync_waits
        );

        rt.set_poison_unzeroed(true);
        let (default, t_default) = traced(|| model.encode(&batch, None, false));
        rt.set_async_encode(true).unwrap();
        let (queued, t_async) = traced(|| model.encode(&batch, None, false));
        rt.set_async_encode(false).unwrap();
        rt.set_poison_unzeroed(false);
        let (default, queued) = (default.unwrap(), queued.unwrap());

        for (what, out, t) in [("default", &default, t_default), ("async_encode", &queued, t_async)] {
            assert_eq!(out.forwards, sync.forwards, "{what}");
            assert_eq!(
                t.sync_waits, 1,
                "{what}: {} forwards made {} host waits, want one for the encode",
                out.forwards, t.sync_waits
            );
            assert_eq!(
                bits(&out.embeddings),
                bits(&sync.embeddings),
                "{what}: queueing the forwards changed an embedding"
            );
        }

        let dim = cfg.embedding_dim as usize;
        for (n, ids) in batch.iter().enumerate() {
            let alone = model.encode(&[ids], None, false).unwrap().embeddings;
            let drift = queued.embeddings[n * dim..(n + 1) * dim]
                .iter()
                .zip(&alone)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                f64::from(drift) <= BATCH_ABS,
                "sequence {n} ({} tokens), queued vs alone: {drift:.3e}",
                ids.len()
            );
        }
    });
}

/// A default runtime (`async_encode` off, as every library caller runs it)
/// still encodes queued: an `encode` that runs several forwards makes one
/// waited commit and one host wait, however many dispatches it has, and
/// leaves the runtime's flag as it found it. Before, `encode` ran every
/// dispatch commit-and-wait unless the caller had set the runtime's global
/// flag (84.9 ms against 7.4 ms for 1x16 tokens on an M5 Pro).
#[test]
fn default_runtime_encode_queues_and_waits_once() {
    assert!(
        std::env::var_os("TESSL_MID_COMMIT").is_none(),
        "TESSL_MID_COMMIT changes the commit and wait counts"
    );
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 31, false);
    let st = open("default_queued", safetensors_bytes(&w, &[]));
    let seq = |n: u32, s: u32| -> Vec<u32> { (0..n).map(|i| (i * 5 + s * 3 + 1) % cfg.vocab).collect() };
    let long = seq(4000, 0);
    let short: Vec<Vec<u32>> = (0..7).map(|s| seq(30, s + 5)).collect();
    let mut batch: Vec<&[u32]> = vec![&long];
    batch.extend(short.iter().map(Vec::as_slice));
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        assert!(
            !rt.async_encode_enabled(),
            "test bug: the default runtime is not synchronous"
        );
        let _ = model.encode(&batch, None, false).unwrap(); // warm pipelines and pools
        let (out, t) = traced(|| model.encode(&batch, None, false).unwrap());
        assert!(out.forwards >= 2, "test bug: {} forwards", out.forwards);
        assert!(t.dispatches > 50, "test bug: only {} dispatches", t.dispatches);
        assert_eq!(
            t.sync_waits, 1,
            "{} forwards, {} dispatches made {} host waits, want one",
            out.forwards, t.dispatches, t.sync_waits
        );
        assert_eq!(
            t.commits, 1,
            "{} dispatches made {} commits, want one command buffer",
            t.dispatches, t.commits
        );
        assert!(!rt.async_encode_enabled(), "encode left the runtime's flag on");
    });
}

/// A full batch of 256 ragged sequences, which splits into several forwards,
/// is one queue: one host wait, a command buffer count bounded by the
/// runtime's per-buffer dispatch cap (100_000, `METAL4_MAX_DISPATCHES_PER_CB`)
/// however many dispatches it makes, and the bits of the serialized run. The
/// per-layer trace of one sequence, which waits at every layer, gives the
/// same bits queued as serialized. No scope outlives either call.
#[test]
fn full_ragged_batch_and_trace_match_serialized_with_bounded_queue() {
    const CAP: u64 = 100_000;
    assert!(
        std::env::var_os("TESSL_MID_COMMIT").is_none(),
        "TESSL_MID_COMMIT changes the commit and wait counts"
    );
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 41, false);
    let st = open("ragged_256", safetensors_bytes(&w, &[]));
    let mut rng = SplitMix::new(99);
    let seqs: Vec<Vec<u32>> = (0..256usize)
        .map(|s| {
            // Lengths from 1 to 150: some forwards hold many short rows, some a few long.
            let n = rng.range(1, 150);
            (0..n).map(|i| ((i * 13 + s * 7 + 1) as u32) % cfg.vocab).collect()
        })
        .collect();
    let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        let _ = model.encode(&batch, None, false).unwrap(); // warm pipelines and pools
        let (queued, t) = traced(|| model.encode(&batch, None, false).unwrap());
        assert_eq!(rt.async_scope_depth(), 0, "encode leaked a scope");
        assert!(!rt.async_encode_enabled());
        assert_eq!(
            t.sync_waits, 1,
            "{} forwards made {} host waits",
            queued.forwards, t.sync_waits
        );
        assert!(
            t.commits <= 1 + t.dispatches / CAP,
            "{} dispatches made {} command buffers, past the {CAP}-dispatch cap",
            t.dispatches,
            t.commits
        );

        rt.set_serialized_dispatch(true).unwrap();
        let serial = model.encode(&batch, None, false).unwrap();
        let serial_trace = model.encode(&batch[..1], None, true).unwrap();
        rt.set_serialized_dispatch(false).unwrap();
        // The two runs cut the batch differently (a queued forward costs 67
        // padded rows, a serialized one 1046), so the same sequence sits in a
        // different GEMM tiling: equal to within `BATCH_ABS`, not bit for bit.
        assert!(
            queued.forwards > serial.forwards,
            "test bug: queued {} forwards, serialized {}",
            queued.forwards,
            serial.forwards
        );
        let drift = queued
            .embeddings
            .iter()
            .zip(&serial.embeddings)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            f64::from(drift) <= BATCH_ABS,
            "256 sequences, queued vs serialized: {drift:.3e}"
        );

        let traced_out = model.encode(&batch[..1], None, true).unwrap();
        assert_eq!(rt.async_scope_depth(), 0);
        assert_eq!(
            traced_out.trace.len(),
            cfg.layers.len() + 1,
            "each layer, then the final norm"
        );
        assert_eq!(traced_out.trace.len(), serial_trace.trace.len());
        for (l, (a, b)) in traced_out.trace.iter().zip(&serial_trace.trace).enumerate() {
            assert_eq!(bits(a), bits(b), "trace entry {l}, queued vs serialized");
        }
        assert_eq!(bits(&traced_out.embeddings), bits(&serial_trace.embeddings));
    });
}

/// The per-layer inputs formed a run of layers per GEMM: every layer in one
/// (the default at these sizes), or runs of 2 and 1 layers under a small
/// `ple_run_bytes`, which read row-offset slices of the projection and column
/// windows past a run's first layer. The embeddings agree within GEMM
/// rounding, and each is the f64 forward's.
fn per_layer_input_runs_agree_at(operands: GemmOperands) {
    let Bounds { emb_abs, emb_cos, .. } = bounds(operands);
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = weights_for(&cfg, 11, operands);
    let st = open("runs", safetensors_bytes(&w, &[]));
    let seqs = sequences(cfg.vocab);
    let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let rows = seqs.len() * seqs.iter().map(Vec::len).max().unwrap();
    let per_layer = rows * cfg.ple_dim as usize * 4;
    let dim = cfg.embedding_dim as usize;
    with_gpu(|rt| {
        let mut model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
        let all = model.encode(&refs, None, false).unwrap().embeddings;
        assert!(model.set_ple_run_bytes(0).is_err());
        for layers_per_run in [2usize, 1] {
            model.set_ple_run_bytes(layers_per_run * per_layer).unwrap();
            let got = model.encode(&refs, None, false).unwrap().embeddings;
            let drift = got.iter().zip(&all).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("runs of {layers_per_run}: vs one run {drift:.3e}");
            assert!(f64::from(drift) <= BATCH_ABS, "runs of {layers_per_run}: {drift:.3e}");
            for (n, ids) in seqs.iter().enumerate() {
                let (_, want) = host_forward(&cfg, &w, ids);
                let e = &got[n * dim..(n + 1) * dim];
                let (err, cos) = (max_abs(e, &want), cosine(e, &want));
                assert!(
                    err <= emb_abs && cos >= emb_cos,
                    "runs of {layers_per_run}, sequence {n}: max abs {err:.3e}, cosine {cos:.9}"
                );
            }
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
    for operands in MODES {
        // Each tensor is held in the dtype it is stored in: f32 projections
        // as f32 under `ExactF32`, bf16 ones as bf16 under `Bf16`.
        let w = weights_for(&cfg, 2026, operands);
        let st = open("drop", safetensors_bytes(&w, &[]));
        with_gpu(|rt| {
            rt.synchronize().unwrap();
            let before = rt.current_allocated_bytes();
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            let loaded = rt.current_allocated_bytes();
            let weights: u64 =
                w.0.iter()
                    .map(|t| t.data.len() as u64 * if t.bf16 { 2 } else { 4 })
                    .sum();
            // Buffers under 16 KiB (the norms) are not charged one by one, so the
            // load is held to half its bytes: enough to show the weights are seen.
            assert!(
                loaded >= before + weights / 2,
                "{operands:?}: loading added {} bytes for {weights} of weights",
                loaded - before
            );
            drop(model);
            rt.synchronize().unwrap();
            let after = rt.current_allocated_bytes();
            assert!(
                after <= before,
                "{operands:?}: {} bytes still allocated after the model dropped",
                after.saturating_sub(before)
            );
        });
    }
}

/// `Bf16` keeps every projection at two bytes on the device where `ExactF32`
/// widens it to four, and nothing else changes: the embedding table (bf16
/// either way), the norms and the scalars are the same bytes. Exact, from the
/// model's own accounting ([`EmbedGemma2Model::weight_bytes`]), and visible in
/// the device's allocation.
#[test]
fn bf16_holds_the_projections_at_half_the_bytes() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = weights_for(&cfg, 2026, GemmOperands::Bf16);
    let st = open("bytes", safetensors_bytes(&w, &[]));
    let projection_elems: u64 =
        w.0.iter()
            .filter(|t| t.bf16 && t.name != "embed_tokens.weight")
            .map(|t| t.data.len() as u64)
            .sum();
    assert!(projection_elems > 0, "test bug: no projection is stored as bf16");
    with_gpu(|rt| {
        let (mut bytes, mut device) = (Vec::new(), Vec::new());
        for operands in MODES {
            rt.synchronize().unwrap();
            let before = rt.current_allocated_bytes();
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            device.push(rt.current_allocated_bytes() - before);
            bytes.push(model.weight_bytes() as u64);
        }
        assert_eq!(
            bytes[0] - bytes[1],
            2 * projection_elems,
            "ExactF32 holds {} bytes of weights, Bf16 {}: want a difference of 2 bytes for each of the {projection_elems} projection elements",
            bytes[0],
            bytes[1]
        );
        assert!(
            device[1] < device[0],
            "the device holds {} bytes after a Bf16 load, {} after an ExactF32 one",
            device[1],
            device[0]
        );
    });
}

/// A `Bf16` load keeps the checkpoint's bf16 projections as they are, so it
/// refuses one stored in another dtype by name instead of rounding it
/// silently; the same file loads at `ExactF32`.
#[test]
fn bf16_load_refuses_projections_that_are_not_bf16() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 2026, false);
    let st = open("f32_proj", safetensors_bytes(&w, &[]));
    with_gpu(|rt| {
        let err = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), GemmOperands::Bf16)
            .err()
            .expect("a Bf16 load of f32 projections succeeded");
        assert!(err.contains("BF16") && err.contains("proj"), "{err}");
        EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), GemmOperands::ExactF32).unwrap();
    });
}

/// The exact-f32 forward needs the runtime's relaxed (tf32-class) f32 GEMMs
/// off and refuses to run otherwise. The bf16 forward runs no f32 GEMM (every
/// projection takes bf16 operands), so the switch cannot touch it: it runs
/// with the switch on, to the bit what it computes with it off.
#[test]
fn relaxed_precision_is_refused_by_exact_f32_and_moves_nothing_at_bf16() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let seqs = sequences(cfg.vocab);
    let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for operands in MODES {
        let w = weights_for(&cfg, 2026, operands);
        let st = open("relaxed", safetensors_bytes(&w, &[]));
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            let exact = model.encode(&refs, None, false).unwrap();
            rt.set_relaxed_precision(true);
            let relaxed = model.encode(&refs, None, false);
            rt.set_relaxed_precision(false);
            match operands {
                GemmOperands::ExactF32 => {
                    let err = relaxed.err().expect("ExactF32 ran with relaxed precision on");
                    assert!(err.contains("exact-f32"), "{err}");
                }
                GemmOperands::Bf16 => assert_eq!(
                    bits(&relaxed.expect("Bf16 refused relaxed precision").embeddings),
                    bits(&exact.embeddings),
                    "relaxed precision changed a bf16 embedding"
                ),
            }
            // The refusal leaves the model usable.
            assert_eq!(
                bits(&model.encode(&refs, None, false).unwrap().embeddings),
                bits(&exact.embeddings)
            );
        });
    }
}

#[test]
fn loader_refuses_tensors_it_would_ignore() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 2026, false);
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

/// `set_max_tokens` takes effect on `encode` and refuses what it cannot
/// honour, leaving the old limit in force when it refuses.
#[test]
fn set_max_tokens_bounds_encode_and_refuses_out_of_range() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = random_weights(&cfg, 2026, false);
    let st = open("max_tokens", safetensors_bytes(&w, &[]));
    let seq = sequences(cfg.vocab).remove(0);
    let n = seq.len() as u32;
    assert!(n > 1, "test bug: need a sequence of at least two tokens");
    with_gpu(|rt| {
        let mut model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        assert_eq!(model.max_tokens(), DEFAULT_MAX_TOKENS);
        let full = model
            .encode(&[&seq], None, false)
            .expect("default limit admits the sequence");

        // Refusals: zero and one past the ceiling; the limit does not move.
        for bad in [0u32, MAX_BATCH_TOKENS as u32 + 1, u32::MAX] {
            let err = model.set_max_tokens(bad).expect_err("out-of-range limit accepted");
            assert!(err.contains("max_tokens must be in 1..="), "{bad}: {err}");
            assert_eq!(
                model.max_tokens(),
                DEFAULT_MAX_TOKENS,
                "a refused set changed the limit"
            );
        }

        // The limit is exactly the longest admitted sequence.
        model.set_max_tokens(n - 1).unwrap();
        assert_eq!(model.max_tokens(), n - 1);
        let err = model
            .encode(&[&seq], None, false)
            .err()
            .expect("over-limit sequence encoded");
        assert!(err.contains("the limit is") && err.contains("set_max_tokens"), "{err}");
        model.set_max_tokens(n).unwrap();
        let again = model
            .encode(&[&seq], None, false)
            .expect("a sequence at the limit encodes");
        assert_eq!(again.embeddings, full.embeddings, "the limit changed the result");

        // The ceiling itself is accepted, and so is 1.
        model.set_max_tokens(MAX_BATCH_TOKENS as u32).unwrap();
        model.set_max_tokens(1).unwrap();
        assert_eq!(model.max_tokens(), 1);
    });
}

/// Adversarial companions to the one-wait encode: with `async_encode` on and
/// unzeroed activations poisoned, a batch that splits into several forwards
/// of different shapes gives bit-identical embeddings to the synchronous
/// encode for a Matryoshka prefix, in a shuffled input order, and when the
/// shapes alternate large, small, large across calls (the shared activations
/// are rewritten by forward n + 1 while forward n may still be queued).
fn async_multi_forward_truncated_shuffled_and_repeated_stay_bitwise_at(operands: GemmOperands) {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = weights_for(&cfg, 23, operands);
    let st = open("async_adversarial", safetensors_bytes(&w, &[]));
    let seq = |n: u32, s: u32| -> Vec<u32> { (0..n).map(|i| (i * 7 + s * 11 + 2) % cfg.vocab).collect() };
    let long = seq(4000, 1);
    let mid: Vec<Vec<u32>> = (0..3).map(|s| seq(300, s + 20)).collect();
    let short: Vec<Vec<u32>> = (0..5).map(|s| seq(1 + s, s + 40)).collect();
    let mut batch: Vec<&[u32]> = vec![&long];
    batch.extend(mid.iter().map(Vec::as_slice));
    batch.extend(short.iter().map(Vec::as_slice));
    let small: Vec<&[u32]> = short.iter().map(Vec::as_slice).collect();
    let mut shuffled = batch.clone();
    shuffled.reverse();
    shuffled.swap(1, 4);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
        let dim = cfg.embedding_dim as usize;
        let trunc = Some(cfg.embedding_dim / 2);
        rt.set_serialized_dispatch(true).unwrap();
        let sync_full = model.encode(&batch, None, false).unwrap();
        let sync_trunc = model.encode(&batch, trunc, false).unwrap();
        let sync_small = model.encode(&small, None, false).unwrap();
        rt.set_serialized_dispatch(false).unwrap();
        assert!(sync_full.forwards >= 3, "test bug: {} forwards", sync_full.forwards);

        rt.set_async_encode(true).unwrap();
        rt.set_poison_unzeroed(true);
        let run = |f: &dyn Fn() -> Result<tessl::embedgemma2::EncodeOutput, String>| f();
        let a_trunc = run(&|| model.encode(&batch, trunc, false));
        let a_small = run(&|| model.encode(&small, None, false));
        let a_full = run(&|| model.encode(&batch, None, false));
        let a_shuf = run(&|| model.encode(&shuffled, None, false));
        rt.set_poison_unzeroed(false);
        rt.set_async_encode(false).unwrap();
        let (a_trunc, a_small, a_full, a_shuf) = (a_trunc.unwrap(), a_small.unwrap(), a_full.unwrap(), a_shuf.unwrap());

        assert_eq!(bits(&a_trunc.embeddings), bits(&sync_trunc.embeddings), "truncated");
        assert_eq!(
            bits(&a_small.embeddings),
            bits(&sync_small.embeddings),
            "small after large"
        );
        assert_eq!(
            bits(&a_full.embeddings),
            bits(&sync_full.embeddings),
            "large after small"
        );
        for row in a_trunc.embeddings.chunks_exact(dim / 2) {
            let n: f32 = row.iter().map(|x| x * x).sum();
            assert!((n - 1.0).abs() < 1e-4, "truncated row norm^2 {n}");
        }
        // Input order is irrelevant: row i of the shuffled run is its sequence's.
        let n = batch.len();
        for (i, s) in shuffled.iter().enumerate() {
            let orig = batch.iter().position(|b| std::ptr::eq(*b, *s)).unwrap();
            assert_eq!(
                bits(&a_shuf.embeddings[i * dim..(i + 1) * dim]),
                bits(&a_full.embeddings[orig * dim..(orig + 1) * dim]),
                "shuffled position {i} of {n}"
            );
        }
    });
}

/// A failed encode (token past the vocab, truncate_dim 0) queues nothing and
/// leaves the model usable: the next encode, async, equals the one before.
fn rejected_encode_leaves_the_async_model_usable_at(operands: GemmOperands) {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let w = weights_for(&cfg, 5, operands);
    let st = open("async_reject", safetensors_bytes(&w, &[]));
    let ok: Vec<u32> = (0..40).map(|i| i % cfg.vocab).collect();
    let bad: Vec<u32> = vec![1, cfg.vocab];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
        rt.set_async_encode(true).unwrap();
        let first = model.encode(&[&ok], None, false);
        let r1 = model.encode(&[&ok, &bad], None, false);
        let r2 = model.encode(&[&ok], Some(0), false);
        let r3 = model.encode(&[&ok, &[]], None, false);
        let again = model.encode(&[&ok], None, false);
        rt.set_async_encode(false).unwrap();
        assert!(r1.is_err() && r2.is_err() && r3.is_err());
        assert_eq!(bits(&first.unwrap().embeddings), bits(&again.unwrap().embeddings));
    });
}

// The invariants above hold at both operand precisions: each test runs once at
// `ExactF32` (under its original name) and once at `Bf16`.
macro_rules! at_both_precisions {
    ($($exact:ident, $bf16:ident => $inner:ident;)*) => {$(
        #[test]
        fn $exact() {
            $inner(GemmOperands::ExactF32);
        }

        #[test]
        fn $bf16() {
            $inner(GemmOperands::Bf16);
        }
    )*};
}

at_both_precisions! {
    encode_reuses_unzeroed_activations, bf16_encode_reuses_unzeroed_activations => encode_reuses_unzeroed_activations_at;
    multi_forward_encode_waits_once_and_matches_synchronous, bf16_multi_forward_encode_waits_once_and_matches_synchronous
        => multi_forward_encode_waits_once_and_matches_synchronous_at;
    per_layer_input_runs_agree, bf16_per_layer_input_runs_agree => per_layer_input_runs_agree_at;
    async_multi_forward_truncated_shuffled_and_repeated_stay_bitwise, bf16_async_multi_forward_truncated_shuffled_and_repeated_stay_bitwise
        => async_multi_forward_truncated_shuffled_and_repeated_stay_bitwise_at;
    rejected_encode_leaves_the_async_model_usable, bf16_rejected_encode_leaves_the_async_model_usable
        => rejected_encode_leaves_the_async_model_usable_at;
}
