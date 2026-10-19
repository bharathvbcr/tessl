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
    load_rt(false).1
}

/// The fixture with its runtime: f32 as `Qwen35Model::load`, or bf16 as the
/// tower a bf16 model is trained from (`load_tower`).
fn load_rt(bf16_tower: bool) -> (std::sync::Arc<GpuRuntime>, Qwen35Model) {
    let dir = fixture();
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let model = if bf16_tower {
        Qwen35Model::load_tower(&rt, &st, "model.", cfg, Precision::Bf16).unwrap()
    } else {
        Qwen35Model::load(&rt, &st, "model.", cfg, Precision::F32).unwrap()
    };
    (rt, model)
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
///
/// Re-pinned (RMSNorm and qk-norm forward/backward to `precise::rsqrt`, owner
/// decision 2026-10-18, card ft-1a0f8c18 AC5): an ulp-level change of the
/// forward norms. Old values: causal 0xfb977a5f6aedd64a, rows
/// 0xd962357482c76de8. The norm rewrite in 4632d2e alone gave causal
/// 0x465a2c57c290769e, rows 0x3aef36370924cc39; the qk-norm kernels in 201bc61
/// moved only the causal digest. Pinned in `--release`.
const GOLDEN_CAUSAL: u64 = 0x06504ffc2068a5b0;
const GOLDEN_ROWS: u64 = 0x3aef36370924cc39;

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

/// Sequence `k` of `len` tokens: ids below the vocabulary of 64, distinct
/// enough between sequences that a leak would show.
fn seq(k: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (i * (5 + 2 * k) + 11 * k + 1) % 64).collect()
}

fn pack(seqs: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut ids = Vec::new();
    let mut cu = vec![0u32];
    for s in seqs {
        ids.extend_from_slice(s);
        cu.push(ids.len() as u32);
    }
    (ids, cu)
}

fn bank_floats(g: &Qwen35Grads) -> Vec<Vec<f64>> {
    all_bits(g)
        .into_iter()
        .map(|v| v.into_iter().map(|b| f64::from(f32::from_bits(b))).collect())
        .collect()
}

/// Largest `|got - want|` over `want`'s largest magnitude, per buffer, worst.
fn worst_rel(got: &Qwen35Grads, want: &Qwen35Grads) -> f64 {
    bank_floats(got)
        .iter()
        .zip(bank_floats(want))
        .map(|(g, w)| {
            let peak = w.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
            g.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / peak
        })
        .fold(0.0, f64::max)
}

/// Bound on a gradient element's difference between the packed step and the
/// sum of the single-sequence steps, relative to the tensor's peak. Both sum
/// the same products; the packed step's weight-gradient GEMMs reduce over all
/// T rows in one pass where the references reduce over each sequence's rows
/// and add the partial sums (and the f32 reductions run in tiles of another
/// shape). Each result differs by at most about `(T + S) * 2^-24` of the sum
/// of the absolute terms; with T = 27, S = 3 that is 1.8e-6 of the terms, and
/// the terms of a gradient with cancellation can exceed its peak by a factor
/// of a few, so the bound is 1e-5 of the peak (as `supervised_rows_are_the_
/// causal_loss_restricted` bounds a sum in another order).
const GRAD_BOUND: f64 = 1e-5;
/// A sequence's mean loss, packed against alone: the same rows through the
/// same kernels (the GEMMs differ only in row count), summed in f64.
const LOSS_BOUND: f64 = 1e-6;

/// The references: every sequence alone as B = 1 (`Supervise::Rows` with
/// the batch's `1 / predictions` scale, accumulated in a bank) and its own
/// `train_step` loss.
fn singles(model: &Qwen35Model, seqs: &[Vec<u32>]) -> (Qwen35Grads, Vec<f64>) {
    let n: usize = seqs.iter().map(|s| s.len() - 1).sum();
    let bank = Qwen35Grads::zeros_like(model).unwrap();
    let mut losses = Vec::new();
    for (k, s) in seqs.iter().enumerate() {
        let pos: Vec<u32> = (0..s.len() as u32 - 1).collect();
        let targets: Vec<u32> = s[1..].to_vec();
        model
            .train_step_into(
                s,
                GemmOperands::ExactF32,
                Supervise::Rows {
                    positions: &pos,
                    targets: &targets,
                    scale: 1.0 / n as f32,
                },
                &bank,
                k > 0,
            )
            .unwrap();
        losses.push(model.train_step(s, GemmOperands::ExactF32).unwrap().loss);
    }
    (bank, losses)
}

#[test]
fn a_ragged_batch_is_each_sequence_alone() {
    let model = load();
    // Lengths off the kernels' 32-row tiles and the GDN chunking, one of
    // them 2 (a single prediction).
    let seqs = [seq(0, 9), seq(1, 2), seq(2, 13), seq(3, 5)];
    let (ids, cu) = pack(&seqs);
    let (want, alone) = singles(&model, &seqs);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let got = model
        .train_step_ragged_into(&ids, &cu, GemmOperands::ExactF32, &bank, false)
        .unwrap();
    let n: usize = seqs.iter().map(|s| s.len() - 1).sum();
    let mean: f64 = seqs
        .iter()
        .zip(&alone)
        .map(|(s, l)| l * (s.len() - 1) as f64)
        .sum::<f64>()
        / n as f64;
    eprintln!("per-sequence {:?} against alone {alone:?}", got.per_sequence);
    for (k, (g, a)) in got.per_sequence.iter().zip(&alone).enumerate() {
        assert!(
            (g - a).abs() <= LOSS_BOUND * a.abs(),
            "sequence {k}: {g} against {a} alone"
        );
    }
    assert!(
        (got.loss - mean).abs() <= LOSS_BOUND * mean.abs(),
        "{} against {mean}",
        got.loss
    );
    let rel = worst_rel(&bank, &want);
    eprintln!("worst gradient rel {rel:.3e}");
    assert!(
        rel <= GRAD_BOUND,
        "gradients are {rel:.3e} of their peak from the singles' sum"
    );
    assert!(bank_floats(&bank).iter().flatten().any(|&x| x != 0.0));
}

#[test]
fn no_token_reads_another_sequence() {
    let model = load();
    let (a, b, c) = (seq(0, 9), seq(1, 7), seq(2, 11));
    // The middle sequence changes; the neighbours' losses must not move by
    // one bit (a leaked state, window or attended key would move them).
    let b2: Vec<u32> = b.iter().map(|&t| (t + 17) % 64).collect();
    let run = |mid: &[u32]| {
        let (ids, cu) = pack(&[a.clone(), mid.to_vec(), c.clone()]);
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        model
            .train_step_ragged_into(&ids, &cu, GemmOperands::ExactF32, &bank, false)
            .unwrap()
            .per_sequence
    };
    let (x, y) = (run(&b), run(&b2));
    assert_ne!(x[1].to_bits(), y[1].to_bits(), "the changed sequence did not change");
    assert_eq!(x[0].to_bits(), y[0].to_bits(), "sequence 0 saw the sequence after it");
    assert_eq!(x[2].to_bits(), y[2].to_bits(), "sequence 2 saw the sequence before it");
}

#[test]
fn a_single_sequence_ragged_step_is_the_row_scored_step() {
    let model = load();
    let s = seq(4, 12);
    let (ids, cu) = pack(std::slice::from_ref(&s));
    let (want, alone) = singles(&model, std::slice::from_ref(&s));
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let got = model
        .train_step_ragged_into(&ids, &cu, GemmOperands::ExactF32, &bank, false)
        .unwrap();
    assert!((got.loss - alone[0]).abs() <= LOSS_BOUND * alone[0].abs());
    // One sequence: the same kernels on the same shapes, the same bits.
    assert_eq!(
        all_bits(&bank),
        all_bits(&want),
        "one packed sequence is not the B = 1 step"
    );
}

#[test]
fn a_ragged_step_accumulates_into_a_bank() {
    let model = load();
    let seqs = [seq(0, 6), seq(1, 8)];
    let (ids, cu) = pack(&seqs);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    model
        .train_step_ragged_into(&ids, &cu, GemmOperands::ExactF32, &bank, false)
        .unwrap();
    let once = bank_floats(&bank);
    model
        .train_step_ragged_into(&ids, &cu, GemmOperands::ExactF32, &bank, true)
        .unwrap();
    let twice = bank_floats(&bank);
    let mut worst = 0.0f64;
    for (o, t) in once.iter().zip(&twice) {
        let peak = o.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
        for (a, b) in o.iter().zip(t) {
            worst = worst.max((2.0 * a - b).abs() / peak);
        }
    }
    assert!(
        worst <= GRAD_BOUND,
        "accumulating the same step twice is {worst:.3e} from double"
    );
}

/// `accumulate = false` writes over the bank, whatever it held: the first
/// sequence's mixer gradients (and the vectors, and the MLP's) replace the
/// bank's, only the later sequences add. Run twice on one bank, and on a bank
/// another ragged step filled first, the result is the zero bank's, bit for
/// bit.
#[test]
fn a_ragged_step_without_accumulate_overwrites_the_bank() {
    let model = load();
    let seqs = [seq(0, 9), seq(1, 6), seq(2, 13)];
    let (ids, cu) = pack(&seqs);
    let step = |bank: &Qwen35Grads, ids: &[u32], cu: &[u32]| {
        model
            .train_step_ragged_into(ids, cu, GemmOperands::ExactF32, bank, false)
            .unwrap();
    };
    let clean = Qwen35Grads::zeros_like(&model).unwrap();
    step(&clean, &ids, &cu);
    let want = all_bits(&clean);
    assert!(want.iter().flatten().any(|&b| f32::from_bits(b) != 0.0));

    // Twice on the same bank: the second overwrites the first.
    step(&clean, &ids, &cu);
    assert_eq!(all_bits(&clean), want, "a second step into the same bank added to it");

    // A bank holding another step's gradients, different in every tensor.
    let dirty = Qwen35Grads::zeros_like(&model).unwrap();
    let other = [seq(5, 7), seq(6, 11)];
    let (oids, ocu) = pack(&other);
    step(&dirty, &oids, &ocu);
    assert_ne!(all_bits(&dirty), want);
    step(&dirty, &ids, &cu);
    assert_eq!(
        all_bits(&dirty),
        want,
        "a bank's old gradients leaked into an overwriting step"
    );
}

/// The step's memory bound counts what a ragged step allocates (each
/// sequence's mixer at its own length, the row-wise buffers once), and the
/// measured peak stays under it, on the f32 model (head in place) and the bf16
/// tower (a bf16 bank, head by chunk).
#[test]
fn a_ragged_steps_peak_stays_under_its_bound() {
    let seqs = [seq(0, 40), seq(1, 17), seq(2, 70), seq(3, 33)];
    let (ids, cu) = pack(&seqs);
    for (bf16, mm) in [(false, GemmOperands::ExactF32), (true, GemmOperands::Bf16)] {
        let (rt, model) = load_rt(bf16);
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        let in_place = !bf16;
        let bound = model.train_step_ragged_bytes(&cu, ids.len(), mm, in_place).unwrap();
        let packed_only = model.train_step_bytes(ids.len() as u32, mm);
        rt.synchronize().unwrap();
        let before = rt.current_allocated_bytes();
        rt.reset_peak_allocated_bytes();
        model.train_step_ragged_into(&ids, &cu, mm, &bank, false).unwrap();
        let peak = rt.peak_allocated_bytes().saturating_sub(before);
        eprintln!("bf16 {bf16}: peak {peak} B, ragged bound {bound} B, one-sequence bound {packed_only} B");
        assert!(peak <= bound, "peak {peak} over the ragged bound {bound}");
        // The bound's last term is the pool's freelist cap, which dwarfs this
        // tiny step; the peak is also under the bound without it, so the
        // step's own bytes are what is counted, not the cap's slack.
        let cap = rt.memory_info().pool_cache_cap as u64;
        assert!(
            peak <= bound - cap,
            "peak {peak} over the bound's own part {}",
            bound - cap
        );
        // Many short sequences cost more than one long one: the bound says so.
        let one = model
            .train_step_ragged_bytes(&[0, ids.len() as u32], ids.len(), mm, in_place)
            .unwrap();
        assert!(bound > one, "{bound} not over the single-sequence bound {one}");
    }
}

/// The bf16 tower on bf16 operands: a bf16 bank holds each gradient rounded
/// to bf16 (relative step 2^-8, error at most 2^-9 of the element) and every
/// delivery rounds again. The packed step rounds each mixer tensor once per
/// sequence (S deliveries) and the rest once; the singles' bank rounds
/// everything S times. Each side's element is within S roundings of its f32
/// sum, plus one for a last-bit f32 difference flipping a rounding, so the two
/// banks differ by at most (2S + 1) * 2^-9 of the tensor's peak. Losses
/// involve no bank: same bound as f32.
#[test]
fn a_ragged_step_on_a_bf16_tower_stays_within_bf16_roundings_of_the_singles() {
    let (_rt, model) = load_rt(true);
    let seqs = [seq(0, 9), seq(1, 6), seq(2, 13)];
    let (ids, cu) = pack(&seqs);
    let n: usize = seqs.iter().map(|s| s.len() - 1).sum();
    let mm = GemmOperands::Bf16;
    let want = Qwen35Grads::zeros_like(&model).unwrap();
    let mut alone = Vec::new();
    for (k, s) in seqs.iter().enumerate() {
        let pos: Vec<u32> = (0..s.len() as u32 - 1).collect();
        model
            .train_step_into(
                s,
                mm,
                Supervise::Rows {
                    positions: &pos,
                    targets: &s[1..],
                    scale: 1.0 / n as f32,
                },
                &want,
                k > 0,
            )
            .unwrap();
        alone.push(model.train_step(s, mm).unwrap().loss);
    }
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let got = model.train_step_ragged_into(&ids, &cu, mm, &bank, false).unwrap();
    for (k, (g, a)) in got.per_sequence.iter().zip(&alone).enumerate() {
        assert!(
            (g - a).abs() <= LOSS_BOUND * a.abs(),
            "sequence {k}: {g} against {a} alone"
        );
    }
    let bound = (2 * seqs.len() + 1) as f64 * 2f64.powi(-9);
    let rel = worst_rel(&bank, &want);
    eprintln!("bf16 tower: worst gradient rel {rel:.3e}, bound {bound:.3e}");
    assert!(rel <= bound, "{rel:.3e} over the derived bound {bound:.3e}");
}

#[test]
fn adversarial_cu_seqlens_are_refused_before_any_gpu_work() {
    use tessl::qwen35_train::check_cu_seqlens;
    let ok = check_cu_seqlens(&[0, 3, 7], 7).unwrap();
    assert_eq!(ok, vec![0..3, 3..7]);
    let bad: &[(&[u32], usize, &str)] = &[
        (&[], 5, "at least 2"),
        (&[0], 0, "at least 2"),
        (&[1, 5], 5, "starts at 1"),
        (&[0, 4, 3, 7], 7, "decreases"),
        (&[0, 3, 3, 7], 7, "is empty"),
        (&[0, 3, 6], 7, "ends at 6"),
        (&[0, 3, 9], 7, "ends at 9"),
        (&[0, 1, 7], 7, "one token"),
        (&[0, u32::MAX], u32::MAX as usize + 1, "overflow u32"),
        (&[0, 3, 5], u32::MAX as usize + 5, "overflow u32"),
    ];
    for (cu, total, needle) in bad {
        let e = check_cu_seqlens(cu, *total).expect_err(needle);
        assert!(e.contains(needle), "{cu:?}: {e:?} lacks {needle:?}");
    }

    // Through the model: refused, and the bank is as it was (zero).
    let model = load();
    let ids = seq(0, 7);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    for (cu, needle) in [
        (vec![], "at least 2"),
        (vec![0, 4, 3, 7], "decreases"),
        (vec![0, 3, 3, 7], "is empty"),
        (vec![1, 7], "starts at 1"),
        (vec![0, 6], "ends at 6"),
    ] {
        let e = model
            .train_step_ragged_into(&ids, &cu, GemmOperands::ExactF32, &bank, true)
            .expect_err(needle);
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    }
    // A token past the vocabulary, and a bank of another shape.
    let (mut oob, cu) = pack(&[seq(0, 5), seq(1, 4)]);
    oob[6] = 64;
    let e = model
        .train_step_ragged_into(&oob, &cu, GemmOperands::ExactF32, &bank, true)
        .unwrap_err();
    assert!(e.contains("token id 64"), "{e:?}");
    let mut short = Qwen35Grads::zeros_like(&model).unwrap();
    short.layers.pop();
    let e = model
        .train_step_ragged_into(&ids, &[0, 7], GemmOperands::ExactF32, &short, true)
        .unwrap_err();
    assert!(e.contains("layers, the model"), "{e:?}");
    assert!(
        bank_floats(&bank).iter().flatten().all(|&x| x == 0.0),
        "a refused step wrote into the bank"
    );
}
