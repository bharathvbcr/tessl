//! `BertSparseModel` end to end on a tiny random checkpoint, built in memory as
//! safetensors bytes: no activation of the forward is read before a kernel of
//! that forward wrote it, and none is zeroed on the host.
//!
//! The batch is large enough that the decoder head runs in several blocks
//! (`HEAD_BLOCK_ROWS / seq` sequences each), so the per-block reset of the
//! pooled rows is exercised more than once.

mod common;

use common::{with_gpu, SplitMix};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tessl::bert::{BertConfig, BertFamily, BertSparseModel, HEAD_BLOCK_ROWS, MAX_BATCH, MAX_BATCH_TOKENS};
use tessl::infer_trace;
use tessl::safetensors::SafeTensors;

const HIDDEN: usize = 64;
const HEADS: usize = 2; // head dim 32
const LAYERS: usize = 2;
const INTER: usize = 128;
const VOCAB: usize = 64;
const POSITIONS: usize = 16;
const TYPES: usize = 2;
// `MAX_SEQ` up to `POSITIONS`, so a head block holds `HEAD_BLOCK_ROWS / MAX_SEQ`
// = 128 sequences and a batch within `MAX_BATCH` still spans two blocks.
const MAX_SEQ: usize = POSITIONS;
const SEQUENCES: usize = MAX_BATCH;

fn config(family: BertFamily) -> BertConfig {
    let json = match family {
        BertFamily::Bert => format!(
            r#"{{"model_type": "bert", "hidden_act": "gelu", "hidden_size": {HIDDEN},
                "num_hidden_layers": {LAYERS}, "num_attention_heads": {HEADS},
                "intermediate_size": {INTER}, "vocab_size": {VOCAB},
                "max_position_embeddings": {POSITIONS}, "type_vocab_size": {TYPES},
                "layer_norm_eps": 1e-12}}"#
        ),
        BertFamily::DistilBert => format!(
            r#"{{"model_type": "distilbert", "activation": "gelu", "dim": {HIDDEN},
                "n_layers": {LAYERS}, "n_heads": {HEADS}, "hidden_dim": {INTER},
                "vocab_size": {VOCAB}, "max_position_embeddings": {POSITIONS}}}"#
        ),
    };
    BertConfig::from_config_json(&json).unwrap()
}

/// Every tensor `BertSparseModel::load` reads, with its shape.
fn tensors(family: BertFamily) -> Vec<(String, Vec<usize>)> {
    let (embed, layer, [q, k, v, o, attn_norm, inter, out, out_norm, transform, transform_norm, bias]) = match family {
        BertFamily::Bert => (
            "bert.embeddings.",
            "bert.encoder.layer.",
            [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
                "attention.output.LayerNorm",
                "intermediate.dense",
                "output.dense",
                "output.LayerNorm",
                "cls.predictions.transform.dense",
                "cls.predictions.transform.LayerNorm",
                "cls.predictions.bias",
            ],
        ),
        BertFamily::DistilBert => (
            "distilbert.embeddings.",
            "distilbert.transformer.layer.",
            [
                "attention.q_lin",
                "attention.k_lin",
                "attention.v_lin",
                "attention.out_lin",
                "sa_layer_norm",
                "ffn.lin1",
                "ffn.lin2",
                "output_layer_norm",
                "vocab_transform",
                "vocab_layer_norm",
                "vocab_projector.bias",
            ],
        ),
    };
    let mut t: Vec<(String, Vec<usize>)> = vec![
        (format!("{embed}word_embeddings.weight"), vec![VOCAB, HIDDEN]),
        (format!("{embed}position_embeddings.weight"), vec![POSITIONS, HIDDEN]),
        (format!("{embed}LayerNorm.weight"), vec![HIDDEN]),
        (format!("{embed}LayerNorm.bias"), vec![HIDDEN]),
    ];
    if family == BertFamily::Bert {
        t.push((format!("{embed}token_type_embeddings.weight"), vec![TYPES, HIDDEN]));
    }
    let dense = |t: &mut Vec<(String, Vec<usize>)>, name: String, inp: usize, outp: usize| {
        t.push((format!("{name}.weight"), vec![outp, inp]));
        t.push((format!("{name}.bias"), vec![outp]));
    };
    let norm = |t: &mut Vec<(String, Vec<usize>)>, name: String| {
        t.push((format!("{name}.weight"), vec![HIDDEN]));
        t.push((format!("{name}.bias"), vec![HIDDEN]));
    };
    for i in 0..LAYERS {
        for name in [q, k, v, o] {
            dense(&mut t, format!("{layer}{i}.{name}"), HIDDEN, HIDDEN);
        }
        norm(&mut t, format!("{layer}{i}.{attn_norm}"));
        dense(&mut t, format!("{layer}{i}.{inter}"), HIDDEN, INTER);
        dense(&mut t, format!("{layer}{i}.{out}"), INTER, HIDDEN);
        norm(&mut t, format!("{layer}{i}.{out_norm}"));
    }
    dense(&mut t, transform.to_string(), HIDDEN, HIDDEN);
    norm(&mut t, transform_norm.to_string());
    t.push((bias.to_string(), vec![VOCAB]));
    t
}

/// Random f32 weights for every tensor, as safetensors bytes. Norm weights
/// sit near 1 so the activations stay well scaled.
fn checkpoint(family: BertFamily, seed: u64) -> SafeTensors {
    checkpoint_with(family, seed, &[])
}

/// [`checkpoint`] plus `extra` tensors `BertSparseModel::load` never reads.
fn checkpoint_with(family: BertFamily, seed: u64, extra: &[(&str, &[usize])]) -> SafeTensors {
    let mut rng = SplitMix::new(seed);
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    let mut all = tensors(family);
    all.extend(extra.iter().map(|&(n, s)| (n.to_string(), s.to_vec())));
    for (i, (name, shape)) in all.iter().enumerate() {
        let n: usize = shape.iter().product();
        let is_norm_weight = name.ends_with("LayerNorm.weight") || name.ends_with("layer_norm.weight");
        let start = data.len();
        for _ in 0..n {
            let x = rng.unit() * 0.1;
            let v = if is_norm_weight { 1.0 + x } else { x };
            data.extend_from_slice(&v.to_le_bytes());
        }
        if i > 0 {
            header.push(',');
        }
        let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
            dims.join(","),
            data.len()
        ));
    }
    header.push('}');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    SafeTensors::from_bytes("bert_tiny", bytes).unwrap()
}

/// `SEQUENCES` token sequences of 3 to `MAX_SEQ` ids; the first is
/// `MAX_SEQ` long, so the batch pads to `MAX_SEQ` rows per sequence.
fn batch(seed: u64) -> Vec<Vec<u32>> {
    let mut rng = SplitMix::new(seed);
    (0..SEQUENCES)
        .map(|i| {
            let len = if i == 0 { MAX_SEQ } else { rng.range(3, MAX_SEQ) };
            (0..len).map(|_| rng.range(0, VOCAB - 1) as u32).collect()
        })
        .collect()
}

#[test]
fn encode_reads_no_unwritten_activation_and_zeroes_nothing_on_the_host() {
    const {
        assert!(
            SEQUENCES > HEAD_BLOCK_ROWS / MAX_SEQ,
            "test bug: the head must run in more than one block"
        );
        assert!(
            SEQUENCES <= MAX_BATCH && SEQUENCES * MAX_SEQ <= MAX_BATCH_TOKENS,
            "test bug: encode refuses a batch past its limits"
        )
    };
    with_gpu(|rt| {
        for family in [BertFamily::Bert, BertFamily::DistilBert] {
            let model = BertSparseModel::load(rt, &checkpoint(family, 7), config(family)).unwrap();
            let seqs = batch(11);
            let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();

            infer_trace::set_enabled(true);
            let s0 = infer_trace::snapshot();
            let clean = model.encode(&refs, false);
            let s1 = infer_trace::snapshot();
            infer_trace::set_enabled(false);
            let clean = clean.unwrap();
            let zeroed = s1.since(&s0).host_zero_bytes;
            assert_eq!(zeroed, 0, "{family:?}: encode zeroed {zeroed} bytes on the host");
            assert_eq!(clean.pooled.len(), SEQUENCES * VOCAB);
            assert!(clean.pooled.iter().all(|v| v.is_finite() && *v >= 0.0));

            rt.set_poison_unzeroed(true);
            let poisoned = model.encode(&refs, false);
            rt.set_poison_unzeroed(false);
            let poisoned = poisoned.unwrap();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&poisoned.pooled),
                bits(&clean.pooled),
                "{family:?}: an unzeroed activation was read before it was written"
            );
        }
    });
}

/// A tensor the model never reads (a deeper layer than the config's, a pooler,
/// a stray bias) is refused by name rather than ignored; the same checkpoint
/// without it loads.
#[test]
fn a_tensor_the_model_never_reads_is_refused_by_name() {
    with_gpu(|rt| {
        for family in [BertFamily::Bert, BertFamily::DistilBert] {
            BertSparseModel::load(rt, &checkpoint(family, 3), config(family)).unwrap();
            let stray: &[&str] = match family {
                BertFamily::Bert => &[
                    "bert.encoder.layer.2.attention.self.query.weight",
                    "bert.pooler.dense.weight",
                    "bert.encoder.layer.0.attention.self.query.extra_bias",
                ],
                BertFamily::DistilBert => &[
                    "distilbert.transformer.layer.2.attention.q_lin.weight",
                    "distilbert.embeddings.stray.weight",
                ],
            };
            for name in stray {
                let bad = checkpoint_with(family, 3, &[(name, &[4])]);
                let err = BertSparseModel::load(rt, &bad, config(family))
                    .err()
                    .unwrap_or_else(|| panic!("{family:?}: {name} was ignored"));
                assert!(err.contains(name), "{family:?}: error does not name {name}: {err}");
            }
        }
    });
}

fn pooled_bits(v: &tessl::bert::SparseOutput) -> Vec<u32> {
    v.pooled.iter().map(|x| x.to_bits()).collect()
}

/// A live `HostMapping` makes `encode` (activations, id/length uploads and the
/// per-block segment upload) return a "busy" error instead of panicking, and
/// leaves nothing behind: the next encode is bit-identical, the runtime is not
/// poisoned, and stale activations are not reused.
#[test]
fn encode_reports_a_busy_runtime_instead_of_panicking_and_leaves_no_state() {
    with_gpu(|rt| {
        for family in [BertFamily::Bert, BertFamily::DistilBert] {
            let model = BertSparseModel::load(rt, &checkpoint(family, 7), config(family)).unwrap();
            let seqs = batch(11);
            let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
            let first = model.encode(&refs, false).unwrap();

            let held = rt.alloc_buffer(64).unwrap();
            let mapping = held.contents_u8();
            let err = catch_unwind(AssertUnwindSafe(|| model.encode(&refs, false).map(|_| ())))
                .unwrap_or_else(|_| panic!("{family:?}: encode panicked on a busy runtime"))
                .unwrap_err();
            assert!(err.contains("busy"), "{family:?}: {err}");
            drop(mapping);

            assert!(rt.synchronize().is_ok(), "{family:?}: the runtime is poisoned");
            let again = model.encode(&refs, false).unwrap();
            assert_eq!(
                pooled_bits(&again),
                pooled_bits(&first),
                "{family:?}: re-encode differs"
            );

            rt.set_poison_unzeroed(true);
            let poisoned = model.encode(&refs, false);
            rt.set_poison_unzeroed(false);
            assert_eq!(
                pooled_bits(&poisoned.unwrap()),
                pooled_bits(&first),
                "{family:?}: stale activations were read"
            );
        }
    });
}

/// With async encode on and a batch already open, an encode waits once per
/// head block (the `rt.synchronize` that frees the logits and pooled buffers
/// for the next block) and for nothing else: the id/length upload of its
/// activations and every block's segment upload add no wait. A waiting upload
/// would make the count exceed the block count.
#[test]
fn async_encode_waits_only_once_per_head_block() {
    assert!(
        std::env::var_os("TESSL_MID_COMMIT").is_none(),
        "TESSL_MID_COMMIT changes the commit and wait counts"
    );
    let head_blocks = SEQUENCES.div_ceil(HEAD_BLOCK_ROWS / MAX_SEQ) as u64;
    assert_eq!(head_blocks, 2, "test bug: the batch should span two head blocks");
    with_gpu(|rt| {
        let family = BertFamily::Bert;
        let model = BertSparseModel::load(rt, &checkpoint(family, 7), config(family)).unwrap();
        let seqs = batch(11);
        let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        // The reference: every dispatch commits and waits, nothing overlaps.
        rt.set_serialized_dispatch(true).unwrap();
        let sync_out = model.encode(&refs, false).unwrap();
        rt.set_serialized_dispatch(false).unwrap();

        // A default runtime (flag off) queues too: the same waits, the same bits.
        infer_trace::set_enabled(true);
        let d0 = infer_trace::snapshot();
        let default_out = model.encode(&refs, false);
        let d1 = infer_trace::snapshot();
        infer_trace::set_enabled(false);
        let default_out = default_out.unwrap();
        assert!(!rt.async_encode_enabled(), "encode left the runtime's flag on");
        assert_eq!(
            d1.since(&d0).sync_waits,
            head_blocks,
            "a default-runtime encode waited more than once per head block"
        );
        assert_eq!(
            pooled_bits(&default_out),
            pooled_bits(&sync_out),
            "default encode differs from the serialized encode"
        );

        // Positive control, then the measured encode, each behind an open batch.
        rt.set_async_encode(true).unwrap();
        let dst = rt.alloc_buffer(16).unwrap();
        rt.upload_u32(&dst, &[1, 2, 3, 4]).unwrap();
        infer_trace::set_enabled(true);
        let c0 = infer_trace::snapshot();
        let control = dst.try_write_u32(&[9, 9, 9, 9]);
        let c1 = infer_trace::snapshot();
        infer_trace::set_enabled(false);
        control.unwrap();
        assert!(
            c1.since(&c0).sync_waits >= 1,
            "positive control: a bound host write counted no wait"
        );

        rt.upload_u32(&dst, &[1, 2, 3, 4]).unwrap();
        infer_trace::set_enabled(true);
        let s0 = infer_trace::snapshot();
        let out = model.encode(&refs, false);
        let s1 = infer_trace::snapshot();
        infer_trace::set_enabled(false);
        rt.synchronize().unwrap();
        rt.set_async_encode(false).unwrap();
        let waits = s1.since(&s0).sync_waits;
        let out = out.unwrap();
        assert_eq!(
            waits, head_blocks,
            "encode waited {waits} times for {head_blocks} head blocks"
        );
        assert_eq!(
            pooled_bits(&out),
            pooled_bits(&sync_out),
            "async encode differs from sync encode"
        );
    });
}

/// The refusals of `encode` before any GPU work, with their exact texts (a
/// characterization: the batch-shape checks are shared with EmbedGemma2's).
#[test]
fn encode_refuses_a_bad_batch_with_exact_texts() {
    const W: &str = "BertSparseModel::encode";
    with_gpu(|rt| {
        let model = BertSparseModel::load(rt, &checkpoint(BertFamily::Bert, 7), config(BertFamily::Bert)).unwrap();
        let err = |b: &[&[u32]], trace: bool| model.encode(b, trace).map(|_| ()).unwrap_err();
        let ok: Vec<u32> = vec![1, 2, 3];
        assert_eq!(err(&[], false), format!("{W}: empty batch"));
        let many: Vec<&[u32]> = vec![ok.as_slice(); MAX_BATCH + 1];
        assert_eq!(
            err(&many, false),
            format!("{W}: {} sequences exceeds the batch limit {MAX_BATCH}", MAX_BATCH + 1)
        );
        assert_eq!(err(&[&ok, &ok], true), format!("{W}: trace takes one sequence, got 2"));
        assert_eq!(err(&[&ok, &[]], false), format!("{W}: sequence 1 is empty"));
        let long = vec![1u32; POSITIONS + 1];
        assert_eq!(
            err(&[&ok, &long], false),
            format!(
                "{W}: sequence 1 has {} tokens; the model has {POSITIONS} positions (truncate it)",
                POSITIONS + 1
            )
        );
        assert_eq!(
            err(&[&[VOCAB as u32]], false),
            format!("{W}: sequence 0: token id {VOCAB} >= vocab {VOCAB}")
        );
    });
}
