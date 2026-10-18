//! Ragged (packed, `cu_seqlens`) Qwen3.5 training steps, and the pins that the
//! existing single-sequence path did not move.
//!
//! Runs on the tiny random fixture of `tests/qwen35_train.rs` (one GDN and
//! one attention layer, tied embeddings).

use std::path::{Path, PathBuf};

use tessl::gemm::GemmOperands;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{MixerGrads, Qwen35Grads, Supervise};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{bf16_bits_to_f32, DType, GpuBuffer, Tensor};
use tessl::GpuRuntime;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn load() -> Qwen35Model {
    let dir = fixture();
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    Qwen35Model::load(&rt, &st, "model.", cfg, Precision::F32).unwrap()
}

fn buf_bits(b: &GpuBuffer) -> Vec<u32> {
    b.read_f32().iter().map(|x| x.to_bits()).collect()
}

fn tensor_bits(t: &Tensor) -> Vec<u32> {
    let n = t.numel();
    match t.dtype {
        DType::BF16 => t.buffer.contents_u16()[..n]
            .iter()
            .map(|&b| bf16_bits_to_f32(b).to_bits())
            .collect(),
        _ => t.buffer.read_f32()[..n].iter().map(|x| x.to_bits()).collect(),
    }
}

/// Every gradient buffer's f32 bits, in a fixed walk of the bank.
fn all_bits(g: &Qwen35Grads) -> Vec<Vec<u32>> {
    let mut v = vec![tensor_bits(&g.embed), buf_bits(&g.final_norm)];
    if let Some(h) = &g.lm_head {
        v.push(tensor_bits(h));
    }
    for l in &g.layers {
        v.extend([buf_bits(&l.input_norm), buf_bits(&l.post_norm)]);
        match &l.mixer {
            MixerGrads::Gdn(m) => v.extend([
                tensor_bits(&m.w_in),
                tensor_bits(&m.w_out),
                buf_bits(&m.conv_w),
                buf_bits(&m.a_log),
                buf_bits(&m.dt_bias),
                buf_bits(&m.norm_w),
            ]),
            MixerGrads::Attn(m) => v.extend([
                tensor_bits(&m.w_in),
                tensor_bits(&m.w_out),
                buf_bits(&m.q_norm),
                buf_bits(&m.k_norm),
            ]),
        }
        v.extend([tensor_bits(&l.gate), tensor_bits(&l.up), tensor_bits(&l.down)]);
    }
    v
}

/// FNV-1a over a loss's bits and every gradient's bits.
fn digest(loss: f64, g: &Qwen35Grads) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |w: u32| {
        for b in w.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    let l = loss.to_bits();
    eat(l as u32);
    eat((l >> 32) as u32);
    for part in all_bits(g) {
        eat(part.len() as u32);
        for w in part {
            eat(w);
        }
    }
    h
}

/// Two fixed sequences (token ids below the fixture's vocabulary of 64).
fn seq_a() -> Vec<u32> {
    (0..13u32).map(|i| (i * 7 + 3) % 64).collect()
}

/// Digests of the single-sequence path, pinned on the code before the ragged
/// entry existed (commit "tests: pin the single-sequence step's bits"):
/// `train_step` on [`seq_a`] (exact f32), and `train_step_into` with
/// `Supervise::Rows` on it. Bit-identical means these never change; they are
/// the bits of this GPU family's kernels, so a different GPU may need them
/// re-pinned from the pre-change commit.
const GOLDEN_CAUSAL: u64 = 0xfb977a5f6aedd64a;
const GOLDEN_ROWS: u64 = 0xd962357482c76de8;

fn causal_digest(model: &Qwen35Model) -> u64 {
    let a = seq_a();
    let s = model.train_step(&a, GemmOperands::ExactF32).unwrap();
    digest(s.loss, &s.grads)
}

fn rows_digest(model: &Qwen35Model) -> u64 {
    let a = seq_a();
    let pos: Vec<u32> = (0..a.len() as u32 - 1).step_by(2).collect();
    let targets: Vec<u32> = pos.iter().map(|&p| a[p as usize + 1]).collect();
    let bank = Qwen35Grads::zeros_like(model).unwrap();
    let loss = model
        .train_step_into(
            &a,
            GemmOperands::ExactF32,
            Supervise::Rows {
                positions: &pos,
                targets: &targets,
                scale: 0.25,
            },
            &bank,
            false,
        )
        .unwrap();
    digest(loss, &bank)
}

#[test]
fn the_single_sequence_step_is_bit_identical_to_its_pin() {
    let model = load();
    let (c, r) = (causal_digest(&model), rows_digest(&model));
    eprintln!("causal digest {c:#018x}, rows digest {r:#018x}");
    assert_eq!(c, GOLDEN_CAUSAL, "train_step moved");
    assert_eq!(r, GOLDEN_ROWS, "train_step_into(Rows) moved");
}
