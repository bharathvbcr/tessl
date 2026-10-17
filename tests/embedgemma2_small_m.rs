//! The text encoder's small-M opt-in (`gemm_small_m`, the parallel split-K for
//! the few-tile exact-f32 GEMMs) at the checkpoint's real GEMM shapes, with
//! synthetic weights: hidden 512, per-layer input 512, intermediate 2048, 4
//! heads of 256 (one global layer of 512, one KV head), so the down GEMM is K
//! 2048 / N 512, the sliding o K 1024 / N 512, the global o K 2048 / N 512 and
//! the per-layer-input GEMMs K 512 / N 512. Two layers (one sliding, one
//! global) and a 1024-token vocabulary keep the checkpoint's GEMMs and drop
//! its depth and embedding table; `tests/embedgemma2_tiny.rs`'s model has K
//! under 512 and never reaches the lane.
//!
//! The question: does a sequence encoded alone (the lane: 10 to 40 rows) stay
//! within the existing alone-vs-batched bound of the same sequence in a batch
//! (200 rows, past the gate: the single dispatch)? The bound is the real
//! checkpoint test's, `BATCH_ABS = 1e-5` of `tests/embedgemma2_model.rs`, fixed
//! here before the first run and not loosened since.
//!
//! The traced child proves both sides are what the question assumes: the
//! parallel kernel dispatches for the alone calls and not once for the batch.

mod common;

use common::embedgemma2::{open, safetensors_bytes, weights_for, PREFIX};
use common::{in_traced_child, with_gpu};
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, PLE_RUN_BYTES};
use tessl::gemm::GemmOperands;
use tessl::runtime::traced_kernel_count;
use tessl::{BufferKind, GpuRuntime};

const BATCH_ABS: f64 = 1e-5;
const PAR: &str = "matmul2d_tensorops_nn_splitk_par_f32";

/// The checkpoint's text shapes with two layers: sliding, then full (global
/// head dimension 512 on one KV head, as `per_layer_config` gives layer 5).
const REAL_SHAPES: &str = r#"{"model_type":"embedding_gemma2","text_config":{
"model_type":"embedding_gemma2_text","attention_bias":false,"hidden_activation":"gelu_pytorch_tanh",
"hidden_size":512,"hidden_size_per_layer_input":512,"intermediate_size":2048,"vocab_size":1024,
"embedding_dim":768,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":256,
"num_hidden_layers":2,"sliding_window":512,"rms_norm_eps":1e-06,
"layer_types":["sliding_attention","full_attention"],
"per_layer_config":{"01":{"head_dim":512,"num_key_value_heads":1}},
"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

/// Five sequences of 10 to 40 tokens, ragged: alone each is 10-40 rows (the
/// lane), together 5 x 40 = 200 padded rows (past it).
fn sequences(vocab: u32) -> Vec<Vec<u32>> {
    [10usize, 16, 23, 31, 40]
        .iter()
        .enumerate()
        .map(|(s, &len)| (0..len as u32).map(|i| (i * 7 + s as u32 * 13 + 3) % vocab).collect())
        .collect()
}

#[test]
fn text_alone_and_batched_stay_within_the_batch_bound_at_real_shapes() {
    let _ = in_traced_child(
        "text_alone_and_batched_stay_within_the_batch_bound_at_real_shapes",
        || {
            let cfg = EmbedGemma2Config::from_config_json(REAL_SHAPES).unwrap();
            let seqs = sequences(cfg.vocab);
            let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
            let dim = cfg.embedding_dim as usize;
            let w = weights_for(&cfg, 2026, GemmOperands::ExactF32);
            let st = open("small_m_real_shapes", safetensors_bytes(&w, &[]));
            with_gpu(|rt| {
                let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), GemmOperands::ExactF32).unwrap();
                let before = traced_kernel_count(PAR);
                let mut alone = Vec::new();
                for r in &refs {
                    let out = model.encode(&[*r], None, false).unwrap();
                    alone.push(out.embeddings);
                }
                let alone_par = traced_kernel_count(PAR) - before;
                let before = traced_kernel_count(PAR);
                let batched = model.encode(&refs, None, false).unwrap();
                assert_eq!(batched.forwards, 1, "test bug: the batch split");
                let batched_par = traced_kernel_count(PAR) - before;
                // 2 layers x (down, o, ple_gate, ple_out) = 8 lane GEMMs per forward.
                assert_eq!(
                    alone_par,
                    8 * refs.len() as u64,
                    "each alone forward runs the lane's 8 GEMMs"
                );
                assert_eq!(batched_par, 0, "the batched forward is past the gate everywhere");
                let mut worst = 0.0f64;
                for (n, e) in alone.iter().enumerate() {
                    let d = max_abs_diff(e, &batched.embeddings[n * dim..(n + 1) * dim]);
                    eprintln!("text {n} ({} tokens): alone vs batched max abs {d:.3e}", seqs[n].len());
                    worst = worst.max(d);
                    assert!(
                        d <= BATCH_ABS,
                        "text {n}: alone vs batched {d:.3e} (bound {BATCH_ABS:e})"
                    );
                }
                // The two routes sum K in different orders, so the bits differ somewhere:
                // were they bit-equal throughout, the test would not be exercising the lane.
                assert!(
                    worst > 0.0,
                    "alone and batched are bit-identical: the lane changed no sum"
                );
            });
        },
    );
}

/// The lane repeats bit for bit: the same sequence alone twice.
#[test]
fn the_text_forward_repeats_bit_for_bit_through_the_lane() {
    let cfg = EmbedGemma2Config::from_config_json(REAL_SHAPES).unwrap();
    let seqs = sequences(cfg.vocab);
    let w = weights_for(&cfg, 2026, GemmOperands::ExactF32);
    let st = open("small_m_real_shapes_repeat", safetensors_bytes(&w, &[]));
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), GemmOperands::ExactF32).unwrap();
        let a = model.encode(&[seqs[1].as_slice()], None, false).unwrap();
        let b = model.encode(&[seqs[1].as_slice()], None, false).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.embeddings), bits(&b.embeddings));
    });
}

/// Device bytes one alone forward of `seq` tokens allocates, itemized as
/// `tests/embedgemma2_memory.rs` does for the exact-f32 mode (the buffers
/// listed on `Acts`, `q|k|v` sharing `gate|up`, the per-layer-input run at the
/// library's bound, the call's inputs and output), as (bytes charged at 16 KiB
/// and up, bytes of the smaller buffers, which bound the total from above).
fn itemized(cfg: &EmbedGemma2Config, seq: usize) -> (u64, u64) {
    let cold = |n: usize| GpuRuntime::allocated_bytes_for(n, BufferKind::Cold);
    let (nb, r) = (1usize, seq);
    let (h, ple, inter, dim) = (
        cfg.hidden as usize,
        cfg.ple_dim as usize,
        cfg.intermediate as usize,
        cfg.embedding_dim as usize,
    );
    let (mut qkv_w, mut q_w, mut kv_w) = (0usize, 0usize, 0usize);
    for s in &cfg.layers {
        let (q, kv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
        qkv_w = qkv_w.max((q + 2 * kv) * d);
        q_w = q_w.max(q * d);
        kv_w = kv_w.max(kv * d);
    }
    let ple_run = (PLE_RUN_BYTES / 4).max(r * ple).min(r * ple * cfg.layers.len());
    let f = 4usize;
    let items = [
        r * h * f,
        r * h * f,
        r * h * f,
        r * qkv_w.max(2 * inter) * f,
        r * q_w * f,
        r * kv_w * f,
        r * kv_w * f,
        r * q_w.max(inter) * f,
        r * h * f,
        ple_run * f,
        r * ple * f,
        r * ple * f,
        nb * h * f,
        r * 4,
        nb * 4,
        nb * 2 * 4,
        nb * dim * 4,
    ];
    let (big, small): (Vec<u64>, Vec<u64>) = items.iter().map(|&n| cold(n)).partition(|&n| n >= 16 << 10);
    (big.iter().sum(), small.iter().sum())
}

/// The opt-in adds no device memory: one alone forward's bytes are exactly the
/// itemized activation set (`tests/embedgemma2_memory.rs`'s accounting, here
/// for the real GEMM shapes), at the row counts where the lane runs (16, 48
/// and 96) as at the two past it (97, 200). The lane's K-partition scratch is
/// `acts.qkv` (idle at the o, down and per-layer-input GEMMs), not a pool
/// allocation: a pooled scratch of 8 slices x rows x 512 f32 took 1.5 MiB at 16
/// rows and 10 MiB at 96 for these two layers (the checkpoint's 24 layers, 12
/// times that), none of it recycled before the next commit.
///
/// Pipelines are compiled on first use and charged to the device outside the
/// pool, so one warm-up forward runs first; the bytes asserted are the second
/// runtime's.
#[test]
fn the_lane_adds_no_device_memory() {
    let cfg = EmbedGemma2Config::from_config_json(REAL_SHAPES).unwrap();
    let w = weights_for(&cfg, 2026, GemmOperands::ExactF32);
    let st = open("small_m_real_shapes_mem", safetensors_bytes(&w, &[]));
    let ids = |len: usize| -> Vec<u32> { (0..len as u32).map(|i| (i * 7 + 3) % cfg.vocab).collect() };
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), GemmOperands::ExactF32).unwrap();
        model.encode(&[ids(16).as_slice()], None, false).unwrap();
    });
    for len in [16usize, 48, 96, 97, 200] {
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), GemmOperands::ExactF32).unwrap();
            rt.synchronize().unwrap();
            let base = rt.current_allocated_bytes();
            model.encode(&[ids(len).as_slice()], None, false).unwrap();
            let got = rt.current_allocated_bytes() - base;
            let (want, slack) = itemized(&cfg, len);
            assert!(
                (want..=want + slack).contains(&got),
                "{len} rows: one encode took {got} bytes, the itemized set is {want} (+ up to {slack} for buffers under 16 KiB)"
            );
        });
    }
}
