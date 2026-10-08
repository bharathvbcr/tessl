//! `Qwen35Model::train_step` against transformers' autograd.
//!
//! `tests/fixtures/qwen35_train/` is a small random `Qwen3_5ForCausalLM` of
//! the 2B's shape family (one GDN and one attention layer, head dims 128 and
//! 256, grouped KV heads, tied embeddings) with its loss and every parameter's
//! gradient from `loss.backward()` in float32, written by
//! `tools/qwen35_ref/make_train_fixture.py tiny`. Each gradient is compared
//! against that parameter's own largest magnitude, through the packing tessl
//! stores the fused projections in.
//!
//! The real 2B is `real_2b_step_matches_transformers` (ignored: it needs the
//! checkpoint and `make_train_fixture.py 2b`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tessl::attn_train::{AttnTrainDims, AttnTrainWorkspace};
use tessl::cross_entropy::{cross_entropy_rows, CeHidden, CeWorkspace, Reduction};
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::qwen35_bwd::{embed_rows_bwd, scatter_add_rows, EmbedBwdWorkspace};
use tessl::qwen35_adamw::{AdamW, AdamWHyper};
use tessl::qwen35_model::{LayerKind, Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{MixerGrads, Qwen35Grads, Supervise, TrainStep};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits, DType, GpuBuffer, Tensor};
use tessl::GpuRuntime;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn npy_f64(path: &Path) -> (Vec<usize>, Vec<f64>) {
    let a = read_npy(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let data = if let Ok(s) = a.f32_slice() {
        s.iter().map(|&x| f64::from(x)).collect()
    } else if let Ok(s) = a.f64_slice() {
        s.to_vec()
    } else if let Ok(s) = a.i64_slice() {
        s.iter().map(|&x| x as f64).collect()
    } else {
        panic!("{}: unexpected dtype", path.display())
    };
    (a.shape, data)
}

fn ids(dir: &Path) -> Vec<u32> {
    npy_f64(&dir.join("ids.npy")).1.iter().map(|&x| x as u32).collect()
}

/// A `[rows, cols]` gradient read back.
fn read(b: &GpuBuffer, n: usize) -> Vec<f64> {
    b.read_f32()[..n].iter().map(|&x| f64::from(x)).collect()
}

fn read_t(t: &Tensor) -> (usize, usize, Vec<f64>) {
    let s = t.shape();
    (s[0], s[1], widened(t).iter().map(|&x| f64::from(x)).collect())
}

/// Columns `[off, off + width)` of a packed `[in, out]` gradient, transposed
/// to torch's `[width, in]`.
fn packed(t: &Tensor, off: usize, width: usize) -> Vec<f64> {
    let (rows, cols, d) = read_t(t);
    let mut out = Vec::with_capacity(width * rows);
    for c in off..off + width {
        for r in 0..rows {
            out.push(d[r * cols + c]);
        }
    }
    out
}

/// Every tessl gradient under its torch parameter name (with `prefix`), in
/// torch's layout.
fn by_name(cfg: &Qwen35Config, g: &Qwen35Grads, prefix: &str) -> Vec<(String, Vec<f64>)> {
    named(cfg, g, prefix, None)
}

/// [`by_name`] for one part only: `Some(l)` is layer `l`'s gradients alone,
/// `Some(usize::MAX)` the embedding and final norm alone; `None` everything.
fn named(cfg: &Qwen35Config, g: &Qwen35Grads, prefix: &str, part: Option<usize>) -> Vec<(String, Vec<f64>)> {
    let (h, i) = (cfg.hidden as usize, cfg.intermediate as usize);
    let mut out = Vec::new();
    if part.is_none() || part == Some(usize::MAX) {
        out.push((format!("{prefix}embed_tokens.weight"), read_t(&g.embed).2));
        out.push((format!("{prefix}norm.weight"), read(&g.final_norm, h)));
    }
    for (l, lg) in g.layers.iter().enumerate() {
        if part.is_some_and(|p| p != l) {
            continue;
        }
        let p = |s: &str| format!("{prefix}layers.{l}.{s}");
        out.push((p("input_layernorm.weight"), read(&lg.input_norm, h)));
        out.push((p("post_attention_layernorm.weight"), read(&lg.post_norm, h)));
        out.push((p("mlp.gate_proj.weight"), packed(&lg.gate, 0, i)));
        out.push((p("mlp.up_proj.weight"), packed(&lg.up, 0, i)));
        out.push((p("mlp.down_proj.weight"), packed(&lg.down, 0, h)));
        match &lg.mixer {
            MixerGrads::Gdn(m) => {
                let gl = cfg.gdn;
                let (cd, vd, hv) = (gl.conv_dim() as usize, gl.value_dim() as usize, gl.v_heads() as usize);
                out.push((p("linear_attn.in_proj_qkv.weight"), packed(&m.w_in, 0, cd)));
                out.push((
                    p("linear_attn.in_proj_z.weight"),
                    packed(&m.w_in, gl.z_off() as usize, vd),
                ));
                out.push((
                    p("linear_attn.in_proj_b.weight"),
                    packed(&m.w_in, gl.b_off() as usize, hv),
                ));
                out.push((
                    p("linear_attn.in_proj_a.weight"),
                    packed(&m.w_in, gl.a_off() as usize, hv),
                ));
                out.push((p("linear_attn.out_proj.weight"), packed(&m.w_out, 0, h)));
                out.push((
                    p("linear_attn.conv1d.weight"),
                    read(&m.conv_w, cd * cfg.conv_kernel as usize),
                ));
                out.push((p("linear_attn.A_log"), read(&m.a_log, hv)));
                out.push((p("linear_attn.dt_bias"), read(&m.dt_bias, hv)));
                out.push((p("linear_attn.norm.weight"), read(&m.norm_w, gl.v_dim() as usize)));
            }
            MixerGrads::Attn(m) => {
                let al = cfg.attn;
                let (q2, kv) = (
                    (2 * al.q_heads() * al.head_dim()) as usize,
                    (al.kv_heads() * al.head_dim()) as usize,
                );
                out.push((p("self_attn.q_proj.weight"), packed(&m.w_in, 0, q2)));
                out.push((p("self_attn.k_proj.weight"), packed(&m.w_in, al.k_off() as usize, kv)));
                out.push((p("self_attn.v_proj.weight"), packed(&m.w_in, al.v_off() as usize, kv)));
                out.push((p("self_attn.o_proj.weight"), packed(&m.w_out, 0, h)));
                out.push((p("self_attn.q_norm.weight"), read(&m.q_norm, al.head_dim() as usize)));
                out.push((p("self_attn.k_norm.weight"), read(&m.k_norm, al.head_dim() as usize)));
            }
        }
    }
    out
}

/// Worst `|got - want| / max|want|` of one parameter.
fn rel(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max) / peak
}

fn load(dir: &Path, prefix: &str, cfg: Qwen35Config, precision: Precision) -> Qwen35Model {
    load_rt(dir, prefix, cfg, precision).1
}

/// [`load`], keeping the runtime for tensors the model reads or writes.
fn load_rt(dir: &Path, prefix: &str, cfg: Qwen35Config, precision: Precision) -> (Arc<GpuRuntime>, Qwen35Model) {
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let model = Qwen35Model::load(&rt, &st, prefix, cfg, precision).unwrap();
    (rt, model)
}

/// The fixture as [`Qwen35Model::load_tower`] loads it (no packed LM head):
/// how a bf16 model is trained.
fn load_tower(dir: &Path, cfg: Qwen35Config, precision: Precision) -> (Arc<GpuRuntime>, Qwen35Model) {
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let model = Qwen35Model::load_tower(&rt, &st, "model.", cfg, precision).unwrap();
    (rt, model)
}

fn tiny_config() -> Qwen35Config {
    Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap()
}

/// The causal-LM loss of the inference forward's logits, in f64 on the host.
fn inference_loss(model: &Qwen35Model, ids: &[u32]) -> f64 {
    let logits = model.forward(ids, false).unwrap().logits;
    let v = model.config().vocab as usize;
    let mut ce = 0.0f64;
    for (t, row) in logits.chunks(v).take(ids.len() - 1).enumerate() {
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let z: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
        ce += m + z.ln() - row[ids[t + 1] as usize] as f64;
    }
    ce / (ids.len() - 1) as f64
}

/// Loss and every gradient against the fixture; returns the worst relative
/// error.
fn compare(dir: &Path, prefix: &str, cfg: &Qwen35Config, step: &TrainStep, loss_bound: f64, bound: f64) -> f64 {
    let (_, want_loss) = npy_f64(&dir.join("loss.npy"));
    let loss_rel = (step.loss - want_loss[0]).abs() / want_loss[0].abs();
    eprintln!("loss {:.8} vs {:.8} (rel {loss_rel:.2e})", step.loss, want_loss[0]);
    assert!(
        loss_rel <= loss_bound,
        "loss {} vs transformers {}",
        step.loss,
        want_loss[0]
    );
    let mut worst = 0.0f64;
    let mut seen = 0;
    for (name, got) in by_name(cfg, &step.grads, prefix) {
        let path = dir.join(format!("grad.{name}.npy"));
        if !path.exists() {
            continue;
        }
        let (_, want) = npy_f64(&path);
        let r = rel(&got, &want);
        eprintln!("{name}: {r:.2e}");
        assert!(r <= bound, "{name}: rel err {r:.3e} > {bound:e}");
        worst = worst.max(r);
        seen += 1;
    }
    let files = std::fs::read_dir(dir)
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().starts_with("grad."))
        .count();
    assert_eq!(seen, files, "every reference gradient must be compared");
    worst
}

#[test]
fn tiny_step_matches_transformers_autograd() {
    let dir = fixture();
    let cfg = tiny_config();
    assert_eq!(cfg.layers, [LayerKind::LinearAttention, LayerKind::FullAttention]);
    let model = load(&dir, "model.", cfg.clone(), Precision::F32);
    let ids = ids(&dir);
    let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    let worst = compare(&dir, "model.", &cfg, &step, 1e-5, 1e-4);
    eprintln!("worst parameter gradient: {worst:.2e}");

    // The training forward is the inference forward: the same loss from
    // the inference logits.
    let ce = inference_loss(&model, &ids);
    assert!(
        (ce - step.loss).abs() <= 1e-5 * ce.abs(),
        "inference loss {ce} vs training loss {}",
        step.loss
    );

    // A second step gives the same bits.
    let again = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    assert_eq!(again.loss.to_bits(), step.loss.to_bits(), "loss changed on a rerun");
    for ((name, a), (_, b)) in by_name(&cfg, &again.grads, "model.")
        .iter()
        .zip(by_name(&cfg, &step.grads, "model."))
    {
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{name} changed on a rerun"
        );
    }
}

/// The step on bf16 GEMM operands (f32 accumulation, f32 weights, activations
/// and gradients) against the same float32 transformers reference. Bounds set
/// before the first run: the loss within 2^-8 relative and every gradient
/// within 2^-5 of its own peak (bf16 rounds each operand by up to 2^-9
/// relative, compounded through the forward, the recompute and the backward's
/// GEMMs). It must not be the exact-f32 step's bits (nothing rounded
/// otherwise), and two bf16 steps must be the same bits.
#[test]
fn tiny_step_on_bf16_operands_stays_near_transformers() {
    let dir = fixture();
    let cfg = tiny_config();
    let model = load(&dir, "model.", cfg.clone(), Precision::F32);
    let ids = ids(&dir);
    let step = model.train_step(&ids, GemmOperands::Bf16).unwrap();
    let worst = compare(&dir, "model.", &cfg, &step, 2f64.powi(-8), 2f64.powi(-5));
    eprintln!("bf16 operands, worst parameter gradient: {worst:.2e}");

    let exact = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    let differs = step.loss.to_bits() != exact.loss.to_bits()
        || by_name(&cfg, &step.grads, "model.")
            .iter()
            .zip(by_name(&cfg, &exact.grads, "model."))
            .any(|((_, a), (_, b))| a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()));
    assert!(
        differs,
        "the bf16-operand step is the exact-f32 step's bits: nothing was rounded"
    );

    let again = model.train_step(&ids, GemmOperands::Bf16).unwrap();
    assert_eq!(
        again.loss.to_bits(),
        step.loss.to_bits(),
        "bf16 loss changed on a rerun"
    );
    for ((name, a), (_, b)) in by_name(&cfg, &again.grads, "model.")
        .iter()
        .zip(by_name(&cfg, &step.grads, "model."))
    {
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{name} changed on a bf16 rerun"
        );
    }
}

/// The step on a bf16-stored model (bf16 matrices, bf16 layer inputs kept
/// for the backward, f32 accumulation and f32 GDN state, softmax and LSE)
/// against the same float32 transformers reference. The fixture is bf16 on
/// disk, so the weights are the f32 model's values exactly; what this adds
/// over bf16 operands is the residual stream rounded at each layer boundary.
/// Bounds set before the first run: the loss within 2^-7 relative and every
/// gradient within 2^-4 of its own peak (the bf16-operand bounds, doubled for
/// the boundary rounding of up to 2^-9 relative per layer); against the f32
/// model's bf16-operand step, every gradient within 2^-5 of its peak. It must
/// not be that step's bits (the stream was rounded), and two steps must be
/// the same bits.
#[test]
fn tiny_step_on_bf16_storage_stays_near_transformers() {
    let dir = fixture();
    let cfg = tiny_config();
    let (_, model) = load_tower(&dir, cfg.clone(), Precision::Bf16);
    let ids = ids(&dir);
    let step = model.train_step(&ids, GemmOperands::Bf16).unwrap();
    let worst = compare(&dir, "model.", &cfg, &step, 2f64.powi(-7), 2f64.powi(-4));
    eprintln!("bf16 storage, worst parameter gradient: {worst:.2e}");

    let f32_model = load(&dir, "model.", cfg.clone(), Precision::F32);
    let operands = f32_model.train_step(&ids, GemmOperands::Bf16).unwrap();
    let (gap, name) = worst_rel(&cfg, &step.grads, &operands.grads);
    eprintln!("bf16 storage vs bf16 operands: worst {gap:.2e} ({name})");
    assert!(gap <= 2f64.powi(-5), "{name}: {gap:.3e} from the bf16-operand step");
    let differs = step.loss.to_bits() != operands.loss.to_bits()
        || by_name(&cfg, &step.grads, "model.")
            .iter()
            .zip(by_name(&cfg, &operands.grads, "model."))
            .any(|((_, a), (_, b))| a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()));
    assert!(differs, "bf16 storage gave the bf16-operand step's bits: the stream was not rounded");

    let again = model.train_step(&ids, GemmOperands::Bf16).unwrap();
    assert_eq!(again.loss.to_bits(), step.loss.to_bits(), "loss changed on a rerun");
    assert_eq!(
        bits(&cfg, &again.grads),
        bits(&cfg, &step.grads),
        "gradients changed on a rerun"
    );
}

/// The f32 values of a bf16 bank tensor, or an f32 one's.
fn widened(t: &Tensor) -> Vec<f32> {
    let n = t.numel();
    match t.dtype {
        DType::BF16 => t.buffer.contents_u16()[..n]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect(),
        _ => t.buffer.read_f32()[..n].to_vec(),
    }
}

/// Every bank tensor of `g` and the f32 gradient it should hold, by name.
fn bank_pairs<'a>(bank: &'a Qwen35Grads, fresh: &'a Qwen35Grads) -> Vec<(String, &'a Tensor, &'a Tensor)> {
    let mut out = vec![("embed".to_string(), &bank.embed, &fresh.embed)];
    for (l, (b, f)) in bank.layers.iter().zip(&fresh.layers).enumerate() {
        for (n, x, y) in [("gate", &b.gate, &f.gate), ("up", &b.up, &f.up), ("down", &b.down, &f.down)] {
            out.push((format!("layers.{l}.{n}"), x, y));
        }
        match (&b.mixer, &f.mixer) {
            (MixerGrads::Gdn(x), MixerGrads::Gdn(y)) => {
                out.push((format!("layers.{l}.w_in"), &x.w_in, &y.w_in));
                out.push((format!("layers.{l}.w_out"), &x.w_out, &y.w_out));
            }
            (MixerGrads::Attn(x), MixerGrads::Attn(y)) => {
                out.push((format!("layers.{l}.w_in"), &x.w_in, &y.w_in));
                out.push((format!("layers.{l}.w_out"), &x.w_out, &y.w_out));
            }
            _ => panic!("layer {l}: mixers differ"),
        }
    }
    out
}

/// A bf16 model's bank holds its matrices' gradients in bf16 and its other
/// gradients in f32. Each delivery rounds once: a step into the bank is the
/// fresh step's f32 gradients rounded to nearest, bit for bit, and a second
/// step accumulated is `bf16(bank + g)` with the add in f32, bit for bit;
/// the f32 parts are the fresh bits, then their f32 sum. `grad_sq_norm` and
/// `read_gradients` read the bank widened.
#[test]
fn a_bf16_bank_rounds_each_delivery_once() {
    let dir = fixture();
    let cfg = tiny_config();
    let (rt, model) = load_tower(&dir, cfg.clone(), Precision::Bf16);
    let ids = ids(&dir);
    let other: Vec<u32> = ids.iter().rev().copied().collect();
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    for (name, b, _) in bank_pairs(&bank, &bank) {
        assert_eq!(b.dtype, DType::BF16, "{name} banks in {:?}", b.dtype);
    }
    let a = model.train_step(&ids, GemmOperands::Bf16).unwrap();
    let b = model.train_step(&other, GemmOperands::Bf16).unwrap();
    model
        .train_step_into(&ids, GemmOperands::Bf16, Supervise::Causal, &bank, false)
        .unwrap();
    for (name, got, want) in bank_pairs(&bank, &a.grads) {
        let want: Vec<u16> = widened(want).iter().map(|&x| f32_to_bf16_bits(x)).collect();
        let got: Vec<u16> = got.buffer.contents_u16()[..got.numel()].to_vec();
        assert!(got == want, "{name}: the bank is not the step's gradient rounded once");
    }
    let sq: f64 = by_name(&cfg, &bank, "")
        .iter()
        .flat_map(|(_, v)| v.iter().map(|x| x * x))
        .sum();
    let got_sq = model.grad_sq_norm(&bank).unwrap();
    assert!((got_sq - sq).abs() <= 1e-5 * sq, "grad_sq_norm {got_sq} vs {sq}");

    model
        .train_step_into(&other, GemmOperands::Bf16, Supervise::Causal, &bank, true)
        .unwrap();
    // The f32 parts: the f32 sum of the two fresh steps, bit for bit.
    let (fa, fb, fbank) = (bits(&cfg, &a.grads), bits(&cfg, &b.grads), bits(&cfg, &bank));
    for (((name, x), (_, y)), (_, z)) in fa.iter().zip(&fb).zip(&fbank) {
        if name.contains("norm") || name.contains("conv1d") || name.contains("A_log") || name.contains("dt_bias") {
            let want: Vec<u32> = x
                .iter()
                .zip(y)
                .map(|(&p, &q)| (f32::from_bits(p) + f32::from_bits(q)).to_bits())
                .collect();
            assert_eq!(z, &want, "{name}: the f32 bank part is not the f32 sum");
        }
    }
    let first: Vec<Vec<u16>> = bank_pairs(&bank, &a.grads)
        .iter()
        .map(|(_, _, f)| widened(f).iter().map(|&x| f32_to_bf16_bits(x)).collect())
        .collect();
    for ((name, got, g2), r1) in bank_pairs(&bank, &b.grads).into_iter().zip(first) {
        let want: Vec<u16> = r1
            .iter()
            .zip(widened(g2))
            .map(|(&r, g)| f32_to_bf16_bits(bf16_bits_to_f32(r) + g))
            .collect();
        let got: Vec<u16> = got.buffer.contents_u16()[..got.numel()].to_vec();
        assert!(got == want, "{name}: the accumulated bank is not bf16(bank + g)");
    }

    // read_gradients widens the bank exactly.
    let table = model.parameter_table().unwrap();
    let ts: Vec<Tensor> = table
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect();
    model.read_gradients(&bank, &ts).unwrap();
    let embed_read = ts[0].read_f32().unwrap();
    assert_eq!(embed_read, widened(&bank.embed), "read_gradients did not widen the bank's embedding");
}

/// A step on a bf16-stored model holds less than the f32 model's at the same
/// length (each layer's kept input is half the bytes), and its measured peak
/// stays under its own bound, which the pre-flight gate trusts.
#[test]
fn a_bf16_step_peaks_under_its_bound_and_below_the_f32_steps() {
    let dir = fixture();
    let cfg = tiny_config();
    let (rt, model) = load_tower(&dir, cfg.clone(), Precision::Bf16);
    let f32_model = load(&dir, "model.", cfg.clone(), Precision::F32);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    for t in [8u32, 64, 512] {
        let bound = model.train_step_bytes(t, GemmOperands::Bf16);
        let f32_bound = f32_model.train_step_bytes(t, GemmOperands::Bf16);
        assert!(bound < f32_bound, "T={t}: bf16 bound {bound} not below f32's {f32_bound}");
        let ids: Vec<u32> = (0..t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
        rt.synchronize().unwrap();
        let before = rt.current_allocated_bytes();
        rt.reset_peak_allocated_bytes();
        model
            .train_step_into(&ids, GemmOperands::Bf16, Supervise::Causal, &bank, false)
            .unwrap();
        let peak = rt.peak_allocated_bytes().saturating_sub(before);
        eprintln!("T={t}: peak {peak} B, bound {bound} B (f32 model's bound {f32_bound} B)");
        assert!(peak <= bound, "T={t}: peak {peak} over the bound {bound}");
    }
}

/// The tiny model 24 layers deep in the 2B's layer pattern
/// (`make_train_fixture.py tiny --layers 24 --out target/qwen35_train_deep`):
/// how gradient agreement with transformers changes with depth alone, both
/// sides f32 and everything else equal. Prints the worst error per layer, for
/// the 1-D tensors (norms, A_log, dt_bias) and the matrices apart.
#[test]
#[ignore]
fn deep_tiny_step_matches_transformers() {
    let dir = std::env::var_os("QWEN35_TRAIN_DEEP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/qwen35_train_deep"));
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let model = load(&dir, "model.", cfg.clone(), Precision::F32);
    let step = model.train_step(&ids(&dir), GemmOperands::ExactF32).unwrap();
    let (_, want_loss) = npy_f64(&dir.join("loss.npy"));
    eprintln!(
        "loss {:.8} vs {:.8} (rel {:.2e})",
        step.loss,
        want_loss[0],
        (step.loss - want_loss[0]).abs() / want_loss[0]
    );
    let mut per_layer = vec![(0.0f64, 0.0f64); cfg.layers.len()];
    let (mut worst, mut worst_scalar) = (0.0f64, 0.0f64);
    for (name, got) in by_name(&cfg, &step.grads, "model.") {
        let r = rel(&got, &npy_f64(&dir.join(format!("grad.{name}.npy"))).1);
        if got.len() == 1 {
            worst_scalar = worst_scalar.max(r);
        } else {
            worst = worst.max(r);
        }
        if let Some(l) = name
            .strip_prefix("model.layers.")
            .and_then(|x| x.split('.').next())
            .and_then(|x| x.parse::<usize>().ok())
        {
            let one_d = got.len() <= cfg.hidden as usize;
            if one_d && r > 5e-5 {
                let want = npy_f64(&dir.join(format!("grad.{name}.npy"))).1;
                let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                eprintln!("  {name}: {r:.2e} ({} elements, max|ref| {peak:.3e})", got.len());
            }
            let slot = &mut per_layer[l];
            if one_d {
                slot.0 = slot.0.max(r)
            } else {
                slot.1 = slot.1.max(r)
            }
        } else {
            eprintln!("{name}: {r:.2e}");
        }
    }
    for (l, (a, b)) in per_layer.iter().enumerate() {
        eprintln!(
            "layer {l:2} ({:?}): 1-D worst {a:.2e}, matrices worst {b:.2e}",
            cfg.layers[l]
        );
    }
    eprintln!("worst parameter gradient: {worst:.2e} (single-element A_log / dt_bias: {worst_scalar:.2e})");
    // Measured before these bounds were written: tensors of more than one
    // element 2.7e-5 at worst (up from ~1e-6 at two layers: drift with depth
    // alone), and the one-head model's single-element A_log / dt_bias
    // gradients 9.6e-4. Those are each one sum over every token of terms of
    // both signs (layer 17's A_log gradient is only 5.3e-4), so relative to
    // themselves they carry the sum's cancellation, which the larger tensors'
    // max-normalized error does not see.
    assert!(worst <= 1e-4, "worst multi-element parameter gradient {worst:.3e}");
    assert!(
        worst_scalar <= 5e-3,
        "worst single-element parameter gradient {worst_scalar:.3e}"
    );
}

#[test]
fn train_step_refuses_what_it_does_not_implement() {
    let dir = fixture();
    let e = |r: Result<TrainStep, String>, needle: &str| {
        let m = r.err().unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    // A bf16 model trains only as a tower (no packed head copy to go stale),
    // and only on bf16 operands (its weights are them).
    let bf16 = load(&dir, "model.", tiny_config(), Precision::Bf16);
    e(bf16.train_step(&[1, 2, 3], GemmOperands::Bf16), "load_tower");
    let tower = load_tower(&dir, tiny_config(), Precision::Bf16).1;
    e(
        tower.train_step(&[1, 2, 3], GemmOperands::ExactF32),
        "trains on GemmOperands::Bf16",
    );
    let model = load(&dir, "model.", tiny_config(), Precision::F32);
    e(model.train_step(&[5], GemmOperands::ExactF32), "at least two tokens");
    e(
        model.train_step(&[5, 64], GemmOperands::ExactF32),
        "token id 64 >= vocab 64",
    );
}

/// A step whose bound ([`Qwen35Model::train_step_bytes`]) does not fit
/// beside what the device has allocated is refused before any GPU work: the
/// runtime is not poisoned, the bank is untouched, and the same step runs
/// once the working set allows it. That bound covers the step's measured
/// peak at several lengths.
#[test]
fn a_step_over_the_working_set_is_refused_before_it_runs() {
    let dir = fixture();
    let (rt, model) = load_rt(&dir, "model.", tiny_config(), Precision::F32);
    let probed = rt.memory_info().recommended_working_set;
    let pool = rt.memory_info().pool_cache_cap as u64;
    let ids = ids(&dir);
    let t = ids.len() as u32;
    let mm = GemmOperands::ExactF32;
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    rt.synchronize().unwrap();
    let need = model.train_step_bytes(t, mm);
    let have = rt.current_allocated_bytes();
    assert!(
        need > pool,
        "the bound {need} B has nothing but the freelist cap {pool} B"
    );

    rt.set_recommended_working_set_for_test(have + need / 2);
    let refused = |r: Result<(), String>, what: &str| {
        let m = r.err().unwrap_or_else(|| panic!("{what}: an over-budget step ran"));
        assert!(
            m.contains("recommended working set") && m.contains("refused"),
            "{what}: {m}"
        );
        assert!(!rt.is_poisoned(), "{what}: the refusal poisoned the runtime");
    };
    refused(
        model
            .train_step_into(&ids, mm, Supervise::Causal, &bank, false)
            .map(|_| ()),
        "train_step_into",
    );
    refused(
        model.train_forward(&ids, mm, Supervise::Causal).map(|_| ()),
        "train_forward",
    );
    refused(model.train_step(&ids, mm).map(|_| ()), "train_step");
    let untouched = bank.final_norm.read_f32();
    assert!(untouched.iter().all(|&x| x == 0.0), "a refused step wrote the bank");

    rt.set_recommended_working_set_for_test(probed);
    for t in [2, ids.len(), 300] {
        let seq: Vec<u32> = (0..t as u32).map(|i| (i * 37 + 5) % 64).collect();
        rt.set_pool_cache_cap_bytes(0);
        rt.set_pool_cache_cap_bytes(pool as usize);
        rt.synchronize().unwrap();
        let before = rt.current_allocated_bytes();
        let need = model.train_step_bytes(t as u32, mm);
        rt.reset_peak_allocated_bytes();
        let loss = model
            .train_step_into(&seq, mm, Supervise::Causal, &bank, false)
            .unwrap_or_else(|e| panic!("T = {t}: {e}"));
        rt.synchronize().unwrap();
        assert!(loss.is_finite(), "T = {t}: loss {loss}");
        let grew = rt.peak_allocated_bytes() - before;
        assert!(grew > 0, "T = {t}: the step allocated nothing");
        assert!(
            grew <= need,
            "T = {t}: the step grew the device by {grew} B, over its bound {need} B"
        );
    }
}

/// Every gradient's f32 bits, by name.
fn bits(cfg: &Qwen35Config, g: &Qwen35Grads) -> Vec<(String, Vec<u32>)> {
    by_name(cfg, g, "")
        .into_iter()
        .map(|(n, v)| (n, v.iter().map(|&x| (x as f32).to_bits()).collect()))
        .collect()
}

/// A bank takes a step's gradients as `train_step` returns them (a copy:
/// the same bits), and with `accumulate` adds the next step's: each value is
/// the f32 sum of the two steps' own gradients, one rounding. Without it the
/// next step writes over the bank. A bank not shaped like the model is
/// refused.
#[test]
fn a_bank_holds_a_steps_gradients_and_accumulates_the_next() {
    let dir = fixture();
    let cfg = tiny_config();
    let model = load(&dir, "model.", cfg.clone(), Precision::F32);
    let a = ids(&dir);
    // A second, shorter sequence: another length exercises another shape.
    let b: Vec<u32> = a.iter().rev().take(a.len() - 3).copied().collect();
    let mm = GemmOperands::ExactF32;
    let fa = model.train_step(&a, mm).unwrap();
    let fb = model.train_step(&b, mm).unwrap();
    let (wa, wb) = (bits(&cfg, &fa.grads), bits(&cfg, &fb.grads));

    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let la = model.train_step_into(&a, mm, Supervise::Causal, &bank, false).unwrap();
    assert_eq!(la.to_bits(), fa.loss.to_bits());
    for ((name, got), (_, want)) in bits(&cfg, &bank).iter().zip(&wa) {
        assert_eq!(got, want, "{name}: the bank is not the step's gradient");
    }

    let lb = model.train_step_into(&b, mm, Supervise::Causal, &bank, true).unwrap();
    assert_eq!(lb.to_bits(), fb.loss.to_bits());
    let mut moved = 0;
    for (((name, got), (_, x)), (_, y)) in bits(&cfg, &bank).iter().zip(&wa).zip(&wb) {
        for (k, ((&g, &x), &y)) in got.iter().zip(x).zip(y).enumerate() {
            let want = f32::from_bits(x) + f32::from_bits(y);
            assert_eq!(g, want.to_bits(), "{name}[{k}]: {} is not {want}", f32::from_bits(g));
            moved += usize::from(y != 0 && g != x);
        }
    }
    assert!(moved > 0, "the second step added nothing");

    model.train_step_into(&b, mm, Supervise::Causal, &bank, false).unwrap();
    for ((name, got), (_, want)) in bits(&cfg, &bank).iter().zip(&wb) {
        assert_eq!(got, want, "{name}: accumulate = false did not write over the bank");
    }

    // A new bank from recycled memory starts at zero: accumulating into it
    // first gives the step's own gradients.
    drop(bank);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    model.train_step_into(&a, mm, Supervise::Causal, &bank, true).unwrap();
    for ((name, got), (_, x)) in bits(&cfg, &bank).iter().zip(&wa) {
        // One add onto zero: -0.0 comes out as +0.0.
        let want: Vec<u32> = x.iter().map(|&b| (0.0f32 + f32::from_bits(b)).to_bits()).collect();
        assert_eq!(got, &want, "{name}: a new bank was not zero");
    }

    let e = |bank: &Qwen35Grads, needle: &str| {
        let m = model
            .train_step_into(&a, mm, Supervise::Causal, bank, true)
            .err()
            .unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    let mut short = Qwen35Grads::zeros_like(&model).unwrap();
    short.layers.pop();
    e(&short, "layers, the model");
    let mut odd = Qwen35Grads::zeros_like(&model).unwrap();
    odd.final_norm = match &odd.layers[0].mixer {
        MixerGrads::Gdn(g) => g.a_log.clone(),
        MixerGrads::Attn(g) => g.q_norm.clone(),
    };
    e(&odd, "the bank's final_norm holds");
}

/// The worst per-parameter `rel` between two sets of gradients, and its name.
fn worst_rel(cfg: &Qwen35Config, got: &Qwen35Grads, want: &Qwen35Grads) -> (f64, String) {
    by_name(cfg, got, "")
        .iter()
        .zip(by_name(cfg, want, ""))
        .map(|((n, g), (_, w))| (rel(g, &w), n.clone()))
        .fold((0.0, String::new()), |a, b| if b.0 > a.0 { b } else { a })
}

/// A step into a fresh bank, returning the loss and the bank.
fn into_fresh(model: &Qwen35Model, ids: &[u32], sup: Supervise<'_>) -> (f64, Qwen35Grads) {
    let bank = Qwen35Grads::zeros_like(model).unwrap();
    let loss = model
        .train_step_into(ids, GemmOperands::ExactF32, sup, &bank, false)
        .unwrap();
    (loss, bank)
}

/// Each position in `p` scored against the token after it, at `scale`.
fn next_token(ids: &[u32], p: &[u32], scale: f32) -> (Vec<u32>, Vec<u32>, f32) {
    (p.to_vec(), p.iter().map(|&i| ids[i as usize + 1]).collect(), scale)
}

fn rows_of(sel: &(Vec<u32>, Vec<u32>, f32)) -> Supervise<'_> {
    Supervise::Rows {
        positions: &sel.0,
        targets: &sel.1,
        scale: sel.2,
    }
}

/// `Supervise::Rows` against what is already checked against transformers
/// (the causal step) and against itself. Bounds set before the first run: a
/// sum in another order, 1e-5 of each parameter's peak and 1e-6 of the loss.
/// - Every position, `scale = 1 / (T - 1)`: the causal step's loss (as a sum)
///   and gradients.
/// - Two disjoint position sets accumulated in a bank: the union's.
/// - Causality: one position scored on the whole sequence is the same
///   position on the sequence cut just after it, loss and gradients. A
///   gradient scattered onto a later row breaks this (the cut sequence has no
///   such row), and the neighbouring position's loss differs, so the position
///   is not off by one.
/// - No position at all: zero loss and zero gradients everywhere.
#[test]
fn supervised_rows_are_the_causal_loss_restricted() {
    let dir = fixture();
    let cfg = tiny_config();
    let model = load(&dir, "model.", cfg.clone(), Precision::F32);
    let mm = GemmOperands::ExactF32;
    let ids = ids(&dir);
    let t = ids.len();
    let n = t - 1;
    let all: Vec<u32> = (0..n as u32).collect();
    let causal = model.train_step(&ids, mm).unwrap();

    let (sum, g) = into_fresh(&model, &ids, rows_of(&next_token(&ids, &all, 1.0 / n as f32)));
    assert!(
        (sum / n as f64 - causal.loss).abs() <= 1e-6 * causal.loss.abs(),
        "{sum} / {n} vs {}",
        causal.loss
    );
    let (w, name) = worst_rel(&cfg, &g, &causal.grads);
    assert!(w <= 1e-5, "every row at 1/(T-1) vs the causal step: {name} {w:.3e}");

    let (evens, odds): (Vec<u32>, Vec<u32>) = all.iter().partition(|&&p| p % 2 == 0);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let le = model
        .train_step_into(&ids, mm, rows_of(&next_token(&ids, &evens, 0.5)), &bank, false)
        .unwrap();
    let lo = model
        .train_step_into(&ids, mm, rows_of(&next_token(&ids, &odds, 0.5)), &bank, true)
        .unwrap();
    let (lu, union) = into_fresh(&model, &ids, rows_of(&next_token(&ids, &all, 0.5)));
    assert!((le + lo - lu).abs() <= 1e-6 * lu.abs(), "{le} + {lo} vs {lu}");
    let (w, name) = worst_rel(&cfg, &bank, &union);
    assert!(w <= 1e-5, "evens + odds vs all: {name} {w:.3e}");

    let p = (t / 2) as u32;
    let one = next_token(&ids, &[p], 1.0);
    let (lf, full) = into_fresh(&model, &ids, rows_of(&one));
    let (lc, cut) = into_fresh(&model, &ids[..p as usize + 1], rows_of(&one));
    assert!((lf - lc).abs() <= 1e-6 * lf.abs(), "{lf} vs {lc}");
    let (w, name) = worst_rel(&cfg, &full, &cut);
    assert!(
        w <= 1e-5,
        "position {p} on the whole sequence vs cut after it: {name} {w:.3e}"
    );
    let (ln, _) = into_fresh(&model, &ids, rows_of(&next_token(&ids, &[p - 1], 1.0)));
    assert_ne!(ln.to_bits(), lf.to_bits());

    let (l0, none) = into_fresh(&model, &ids, rows_of(&next_token(&ids, &[], 1.0)));
    assert_eq!(l0, 0.0);
    for (name, v) in by_name(&cfg, &none, "") {
        assert!(
            v.iter().all(|&x| x == 0.0),
            "{name}: a step that scores nothing has a gradient"
        );
    }
}

/// The tied embedding `[vocab, hidden]` read back.
fn embed_weight(rt: &Arc<GpuRuntime>, model: &Qwen35Model) -> Vec<f64> {
    let table = model.parameter_table().unwrap();
    let ts: Vec<Tensor> = table
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect();
    model.read_parameters(&ts).unwrap();
    let i = table.iter().position(|p| p.name == "embed_tokens.weight").unwrap();
    ts[i].read_f32().unwrap().iter().map(|&x| f64::from(x)).collect()
}

/// A loss outside tessl: `PendingStep::hidden` gives it the final norm's
/// output, and `train_backward_into` takes its gradient there. The outside
/// loss here is the cross-entropy of the tied head on the host, in f64, so
/// it must be what `Supervise::Rows` computes inside tessl: the same loss
/// from the rows `hidden` returned (so they are the right rows), and, with
/// the host's `dh` fed back on a step that scores nothing plus the host's
/// head gradient `dlogitsᵀ h` on the embedding, the same gradients. Bounds
/// set before the first run: 1e-5 of the loss and of each parameter's peak.
/// A pending step is refused by another model, and a misshapen or repeating
/// `dh` is refused before anything runs.
#[test]
fn a_loss_outside_tessl_flows_back_through_the_hidden_states() {
    let dir = fixture();
    let cfg = tiny_config();
    let (rt, model) = load_rt(&dir, "model.", cfg.clone(), Precision::F32);
    let mm = GemmOperands::ExactF32;
    let ids = ids(&dir);
    let (h, vocab) = (cfg.hidden as usize, cfg.vocab as usize);
    let sel = next_token(&ids, &[1, 4, (ids.len() - 2) as u32], 0.25);
    let (want_loss, want) = into_fresh(&model, &ids, rows_of(&sel));

    let p = model
        .train_forward(&ids, mm, rows_of(&next_token(&ids, &[], 1.0)))
        .unwrap();
    assert_eq!(p.loss(), 0.0);
    let rows = rt.alloc_tensor_f32(&[sel.0.len(), h]).unwrap();
    p.hidden(&sel.0, &rows).unwrap();
    let hs: Vec<f64> = rows.read_f32().unwrap().iter().map(|&x| f64::from(x)).collect();
    let w = embed_weight(&rt, &model);

    // Host cross-entropy of the tied head over the selected rows.
    let (mut loss, mut dh, mut dw_head) = (0.0f64, vec![0.0f64; sel.0.len() * h], vec![0.0f64; vocab * h]);
    for (i, &target) in sel.1.iter().enumerate() {
        let hi = &hs[i * h..(i + 1) * h];
        let logits: Vec<f64> = (0..vocab).map(|v| (0..h).map(|c| hi[c] * w[v * h + c]).sum()).collect();
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = logits.iter().map(|l| (l - m).exp()).sum();
        loss += m + z.ln() - logits[target as usize];
        for v in 0..vocab {
            let d = f64::from(sel.2) * ((logits[v] - m).exp() / z - f64::from(u8::from(v == target as usize)));
            for c in 0..h {
                dh[i * h + c] += d * w[v * h + c];
                dw_head[v * h + c] += d * hi[c];
            }
        }
    }
    assert!(
        (loss - want_loss).abs() <= 1e-5 * want_loss.abs(),
        "host {loss} vs tessl {want_loss}"
    );

    let dh_t = rt.alloc_tensor_f32(&[sel.0.len(), h]).unwrap();
    dh_t.buffer.write_f32(&dh.iter().map(|&x| x as f32).collect::<Vec<_>>());
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    model
        .train_backward_into(p, Some((&sel.0, &dh_t)), &bank, false)
        .unwrap();
    let got = by_name(&cfg, &bank, "");
    for ((name, g), (_, wv)) in got.iter().zip(by_name(&cfg, &want, "")) {
        let g: Vec<f64> = if name == "embed_tokens.weight" {
            g.iter().zip(&dw_head).map(|(a, b)| a + b).collect()
        } else {
            g.clone()
        };
        let r = rel(&g, &wv);
        assert!(r <= 1e-5, "{name}: {r:.3e}");
    }

    // Refusals, each before the step runs.
    let other = load(&dir, "model.", cfg.clone(), Precision::F32);
    let e = |p, dh: Option<(&[u32], &Tensor)>, needle: &str| {
        let m = model
            .train_backward_into(p, dh, &bank, false)
            .err()
            .unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    let fresh = || model.train_forward(&ids, mm, Supervise::Causal).unwrap();
    e(
        other.train_forward(&ids, mm, Supervise::Causal).unwrap(),
        None,
        "another model's",
    );
    e(fresh(), Some((&[1, 1, 2], &dh_t)), "position 1 appears twice");
    e(fresh(), Some((&[1, 2], &dh_t)), "src must be f32 [2, ");
    let bad = rt.alloc_tensor_f32(&[3, h]).unwrap();
    let t = ids.len() as u32;
    e(fresh(), Some((&[1, 2, t], &bad)), &format!("position {t} >= {t} rows"));
    let m = fresh()
        .hidden(&[t], &rt.alloc_tensor_f32(&[1, h]).unwrap())
        .unwrap_err();
    assert!(m.contains(&format!("position {t} >= {t} tokens")), "{m}");
}

/// The backward rebuilds each layer from the weights it finds, so weights
/// written after the forward would give gradients of a function the forward
/// never evaluated. Both writers are refused, even one that writes the
/// values already there; a forward taken after the write still runs.
#[test]
fn weights_written_between_forward_and_backward_are_refused() {
    let dir = fixture();
    let cfg = tiny_config();
    let (rt, model) = load_rt(&dir, "model.", cfg, Precision::F32);
    let mm = GemmOperands::ExactF32;
    let ids = ids(&dir);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let refused = |p, writer: &str| {
        let m = model
            .train_backward_into(p, None, &bank, false)
            .err()
            .unwrap_or_else(|| panic!("a backward after {writer} was accepted"));
        assert!(
            m.contains("parameters were written after the step's forward"),
            "{writer}: {m}"
        );
    };

    let p = model.train_forward(&ids, mm, Supervise::Causal).unwrap();
    let table = model.parameter_table().unwrap();
    let values: Vec<Tensor> = table
        .iter()
        .map(|i| rt.alloc_tensor_f32(&i.storage_shape()).unwrap())
        .collect();
    model.read_parameters(&values).unwrap();
    model.write_parameters(&values).unwrap();
    refused(p, "write_parameters");

    let p = model.train_forward(&ids, mm, Supervise::Causal).unwrap();
    let step = model.train_step(&ids, mm).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let wd = model.default_weight_decay(0.0).unwrap();
    model
        .adamw_step(&step.grads, &mut state, &AdamWHyper::default(), &wd)
        .unwrap();
    refused(p, "adamw_step");

    let p = model.train_forward(&ids, mm, Supervise::Causal).unwrap();
    model
        .train_backward_into(p, None, &bank, false)
        .expect("a forward after the write sees the new weights");
}

/// A bank buffer allocated on another runtime is not in this one's residency
/// set: binding it is a fault, not an error, so both bank entry points refuse
/// it before anything runs.
#[test]
fn check_bank_refuses_a_foreign_runtime() {
    let dir = fixture();
    let cfg = tiny_config();
    let (rt, model) = load_rt(&dir, "model.", cfg, Precision::F32);
    let other = GpuRuntime::new().unwrap();
    let mm = GemmOperands::ExactF32;
    let ids = ids(&dir);
    let refused = |r: Result<(), String>, what: &str| {
        let m = r.err().unwrap_or_else(|| panic!("{what}: a foreign bank was accepted"));
        assert!(m.contains("another runtime"), "{what}: {m}");
        assert_eq!(rt.take_dispatch_count(), 0, "{what} encoded work before refusing");
    };

    let mut top = Qwen35Grads::zeros_like(&model).unwrap();
    top.final_norm = other.alloc_buffer(top.final_norm.nbytes()).unwrap();
    let mut layer = Qwen35Grads::zeros_like(&model).unwrap();
    let l = &mut layer.layers[0];
    l.down = other.alloc_tensor_f32(l.down.shape()).unwrap();
    for bank in [&top, &layer] {
        let _ = rt.take_dispatch_count();
        refused(
            model
                .train_step_into(&ids, mm, Supervise::Causal, bank, false)
                .map(|_| ()),
            "train_step_into",
        );
        let p = model.train_forward(&ids, mm, Supervise::Causal).unwrap();
        let _ = rt.take_dispatch_count();
        refused(model.train_backward_into(p, None, bank, false), "train_backward_into");
    }
}

/// `PendingStep::hidden`'s `out` is written by a kernel on the step's runtime.
#[test]
fn pending_hidden_refuses_a_foreign_runtime() {
    let dir = fixture();
    let cfg = tiny_config();
    let (rt, model) = load_rt(&dir, "model.", cfg.clone(), Precision::F32);
    let other = GpuRuntime::new().unwrap();
    let ids = ids(&dir);
    let p = model
        .train_forward(&ids, GemmOperands::ExactF32, Supervise::Causal)
        .unwrap();
    let out = other.alloc_tensor_f32(&[1, cfg.hidden as usize]).unwrap();
    let _ = rt.take_dispatch_count();
    let m = p.hidden(&[0], &out).unwrap_err();
    assert!(m.contains("another runtime"), "{m}");
    assert_eq!(rt.take_dispatch_count(), 0, "hidden encoded work before refusing");
}

/// A foreign `dh` is refused by the backward's checks, before the step is
/// consumed by anything that runs.
#[test]
fn backward_dh_refuses_a_foreign_runtime() {
    let dir = fixture();
    let cfg = tiny_config();
    let (rt, model) = load_rt(&dir, "model.", cfg.clone(), Precision::F32);
    let other = GpuRuntime::new().unwrap();
    let ids = ids(&dir);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let p = model
        .train_forward(&ids, GemmOperands::ExactF32, Supervise::Causal)
        .unwrap();
    let dh = other.alloc_tensor_f32(&[2, cfg.hidden as usize]).unwrap();
    let _ = rt.take_dispatch_count();
    let m = model
        .train_backward_into(p, Some((&[0, 1], &dh)), &bank, false)
        .unwrap_err();
    assert!(m.contains("dh: buffer belongs to another runtime"), "{m}");
    assert_eq!(rt.take_dispatch_count(), 0, "the backward encoded work before refusing");
}

/// A step that scores nothing in tessl (its embedding gradient goes straight
/// into the bank): accumulated after a scored step, every gradient is the
/// f32 sum of the two steps taken apart, the embedding's included (equal as
/// values: an add onto zero may turn -0.0 into +0.0); without accumulate,
/// into a bank already holding a step, the embedding is zeroed first. The
/// values do not show which path ran: the fresh-tensor path gives the same
/// ones, and the saving is memory, not a number.
#[test]
fn a_step_that_scores_nothing_accumulates_like_any_other() {
    let dir = fixture();
    let cfg = tiny_config();
    let (rt, model) = load_rt(&dir, "model.", cfg.clone(), Precision::F32);
    let mm = GemmOperands::ExactF32;
    let ids = ids(&dir);
    let h = cfg.hidden as usize;
    let at = [2u32, 9, 20];
    let dh = rt.alloc_tensor_f32(&[at.len(), h]).unwrap();
    dh.buffer.write_f32(
        &(0..at.len() * h)
            .map(|i| ((i % 11) as f32 - 5.0) * 1e-2)
            .collect::<Vec<_>>(),
    );
    let nothing = next_token(&ids, &[], 1.0);

    let (_, first) = into_fresh(&model, &ids, Supervise::Causal);
    // Into a bank that already holds a step's gradients: without accumulate
    // its embedding must be zeroed first, not added onto.
    let second = Qwen35Grads::zeros_like(&model).unwrap();
    model
        .train_step_into(&ids, mm, Supervise::Causal, &second, false)
        .unwrap();
    let p = model.train_forward(&ids[..30], mm, rows_of(&nothing)).unwrap();
    model.train_backward_into(p, Some((&at, &dh)), &second, false).unwrap();

    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    model
        .train_step_into(&ids, mm, Supervise::Causal, &bank, false)
        .unwrap();
    let p = model.train_forward(&ids[..30], mm, rows_of(&nothing)).unwrap();
    model.train_backward_into(p, Some((&at, &dh)), &bank, true).unwrap();

    let mut moved = false;
    for (((name, got), (_, x)), (_, y)) in bits(&cfg, &bank)
        .iter()
        .zip(bits(&cfg, &first))
        .zip(bits(&cfg, &second))
    {
        for (k, ((&g, &x), &y)) in got.iter().zip(&x).zip(&y).enumerate() {
            let want = f32::from_bits(x) + f32::from_bits(y);
            assert_eq!(f32::from_bits(g), want, "{name}[{k}]");
            moved |= name == "embed_tokens.weight" && y != 0 && g != x;
        }
    }
    assert!(moved, "the second step added nothing to the embedding");
}

/// `Supervise::Rows` refuses what it cannot score, before anything runs.
#[test]
fn supervised_rows_refuse_bad_selections() {
    let dir = fixture();
    let model = load(&dir, "model.", tiny_config(), Precision::F32);
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let ids = [1u32, 2, 3, 4];
    let e = |positions: &[u32], targets: &[u32], scale: f32, needle: &str| {
        let sup = Supervise::Rows {
            positions,
            targets,
            scale,
        };
        let m = model
            .train_step_into(&ids, GemmOperands::ExactF32, sup, &bank, false)
            .err()
            .unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    e(&[0, 1], &[2], 1.0, "2 positions but 1 targets");
    e(&[4], &[2], 1.0, "position 4 >= 4 tokens");
    e(&[1, 1], &[2, 3], 1.0, "position 1 is supervised twice");
    e(&[1], &[64], 1.0, "target 64 >= vocab 64");
    e(&[1], &[2], f32::NAN, "scale NaN must be finite");
    // One token is enough when the target is given.
    let sup = Supervise::Rows {
        positions: &[0],
        targets: &[7],
        scale: 1.0,
    };
    model
        .train_step_into(&ids[..1], GemmOperands::ExactF32, sup, &bank, false)
        .unwrap();
}

/// A runtime that another host mapping holds (busy), or that an earlier
/// failure poisoned, is an `Err` from every training entry point, never a
/// panic: a trainer calls these directly, with no `catch_unwind` between it
/// and them, so a panic here takes the training process down.
#[test]
fn a_busy_or_poisoned_runtime_is_an_error_from_every_training_entry_point() {
    let dir = fixture();
    let ids = ids(&dir);
    let mm = GemmOperands::ExactF32;
    for (state, needle) in [("busy", "busy"), ("poisoned", "poisoned")] {
        let cfg = tiny_config();
        let (rt, model) = load_rt(&dir, "model.", cfg.clone(), Precision::F32);
        let (h, vocab) = (cfg.hidden as usize, cfg.vocab as usize);
        // Made while the runtime is free: what the entry points take.
        let pending = model.train_forward(&ids, mm, Supervise::Causal).unwrap();
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        let rows = rt.alloc_tensor_f32(&[1, h]).unwrap();
        let dh = rt.alloc_tensor_f32(&[1, h]).unwrap();
        let dw = rt.alloc_tensor_f32(&[vocab, h]).unwrap();
        let dst = rt.alloc_tensor_f32(&[ids.len(), h]).unwrap();
        let ce_ws = CeWorkspace::new(&rt, 1, cfg.hidden, cfg.vocab, DType::F32).unwrap();
        let emb_ws = EmbedBwdWorkspace::new(&rt, ids.len() as u32).unwrap();

        let held = rt.alloc_buffer(4).unwrap();
        let _mapping = held.try_contents_u32().unwrap();
        if state == "poisoned" {
            rt.poison_as_shared_event_timeout_for_test();
        }
        let refused = |what: &str, r: Result<(), String>| {
            let e = r
                .err()
                .unwrap_or_else(|| panic!("{what} on a {state} runtime: accepted"));
            assert!(
                e.contains(needle),
                "{what} on a {state} runtime: {e:?} lacks {needle:?}"
            );
            // How a trainer tells the two apart without matching the string:
            // busy passes once the other access ends, poisoned is permanent.
            assert_eq!(rt.is_poisoned(), state == "poisoned", "{what} on a {state} runtime");
        };
        refused("Qwen35Grads::zeros_like", Qwen35Grads::zeros_like(&model).map(drop));
        refused("AdamW::new", AdamW::new(&model).map(drop));
        refused("train_step", model.train_step(&ids, mm).map(drop));
        refused(
            "train_step_into",
            model
                .train_step_into(&ids, mm, Supervise::Causal, &bank, false)
                .map(drop),
        );
        refused(
            "train_forward",
            model.train_forward(&ids, mm, Supervise::Causal).map(drop),
        );
        refused("PendingStep::hidden", pending.hidden(&[0], &rows));
        refused(
            "train_backward_into",
            model.train_backward_into(pending, Some((&[0], &dh)), &bank, false),
        );
        let dims = AttnTrainDims {
            batch: 1,
            seq: ids.len() as u32,
            q_heads: 2,
            kv_heads: 1,
            scale: 0.0625,
        };
        refused("AttnTrainWorkspace::new", AttnTrainWorkspace::new(&rt, dims).map(drop));
        refused(
            "cross_entropy_rows",
            cross_entropy_rows(
                &rt,
                CeHidden { rows: &rows, off: 0 },
                &dw,
                &[0],
                &[1],
                Reduction::Sum,
                mm,
                &ce_ws,
                None,
            )
            .map(drop),
        );
        refused(
            "embed_rows_bwd",
            embed_rows_bwd(&rt, &ids, &dst.buffer, &dw.buffer, cfg.vocab, cfg.hidden, &emb_ws),
        );
        refused("scatter_add_rows", scatter_add_rows(&rt, &dh, &[0], &dst));
    }
}

/// The 2B reference directory (`make_train_fixture.py 2b`), the model loaded
/// in f32 from `QWEN35_2B_SAFETENSORS`, and its config.
fn real_2b() -> (PathBuf, Qwen35Config, Qwen35Model) {
    let dir = std::env::var_os("QWEN35_TRAIN_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/qwen35_train_ref"));
    let st_path = PathBuf::from(
        std::env::var("QWEN35_2B_SAFETENSORS")
            .expect("set QWEN35_2B_SAFETENSORS to the Qwen3.5-2B-Base .safetensors file"),
    );
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&st_path).unwrap();
    let cfg = Qwen35Config::qwen35_2b().unwrap();
    let model = Qwen35Model::load(&rt, &st, "model.language_model.", cfg.clone(), Precision::F32).unwrap();
    (dir, cfg, model)
}

/// Every reference gradient in `dir` against `step`'s, each as
/// [`rel`], printed before anything is asserted; returns them and the worst.
fn real_2b_results(dir: &Path, cfg: &Qwen35Config, step: &TrainStep) -> (Vec<(String, f64)>, f64) {
    let mut results = Vec::new();
    // The embedding: only the rows the reference kept.
    let (_, rows) = npy_f64(&dir.join("embed_rows.npy"));
    let (_, want) = npy_f64(&dir.join("grad.model.embed_tokens.weight.rows.npy"));
    let (_, h, all) = read_t(&step.grads.embed);
    let got: Vec<f64> = rows.iter().flat_map(|&r| all[r as usize * h..][..h].to_vec()).collect();
    results.push(("model.embed_tokens.weight (kept rows)".to_string(), rel(&got, &want)));
    // transformers names the text tower's parameters `model.*`; the
    // checkpoint stores them under `model.language_model.*`.
    for (name, got) in by_name(cfg, &step.grads, "model.") {
        let path = dir.join(format!("grad.{name}.npy"));
        if path.exists() {
            results.push((name, rel(&got, &npy_f64(&path).1)));
        }
    }
    for (name, r) in &results {
        eprintln!("{name}: {r:.2e}");
    }
    let files = std::fs::read_dir(dir)
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().starts_with("grad."))
        .count();
    assert_eq!(results.len(), files, "every reference gradient must be compared");
    let worst = results.iter().map(|(_, r)| *r).fold(0.0, f64::max);
    eprintln!(
        "worst parameter gradient: {worst:.2e} over {} parameters",
        results.len()
    );
    (results, worst)
}

/// Qwen3.5-2B-Base: `python3 tools/qwen35_ref/make_train_fixture.py 2b`
/// first (see its docstring for memory), then this. Compares the loss, every
/// 1-D and conv parameter, layers 0 and 3 in full, and the embedding rows
/// the reference kept.
#[test]
#[ignore]
fn real_2b_step_matches_transformers() {
    let (dir, cfg, model) = real_2b();
    let ids = ids(&dir);
    let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    let infer = inference_loss(&model, &ids);
    let (_, want_loss) = npy_f64(&dir.join("loss.npy"));
    let want_loss = want_loss[0];
    let r_train = (step.loss - want_loss).abs() / want_loss;
    let r_infer = (infer - want_loss).abs() / want_loss;
    let r_self = (step.loss - infer).abs() / infer;
    eprintln!(
        "loss: train {:.8}, inference {infer:.8}, transformers {want_loss:.8}",
        step.loss
    );
    eprintln!("      train vs transformers {r_train:.2e}, inference vs transformers {r_infer:.2e}, train vs inference {r_self:.2e}");
    let (results, _) = real_2b_results(&dir, &cfg, &step);
    // These bounds were set after the first run, from what it showed: the
    // training loss is the inference forward's (1.7e-7), which differs from
    // transformers' fp32 loss by 4.6e-5 because the two f32 forwards differ
    // (logits by 1.9e-6 of the largest, tests/qwen35_model.rs). Gradients of
    // two slightly different functions differ more, up to 3.9e-3 of a
    // parameter's largest here, against transformers' own run-to-run 3.7e-5
    // (tools/qwen35_ref/train_noise_floor.py). That the difference is the
    // forward's and not the backward's is qwen35_train's unit test
    // `real_2b_gradients_are_those_of_tessls_forward`: finite differences of
    // tessl's own loss land on tessl's gradients, not transformers'.
    assert!(
        r_self <= 1e-5,
        "training loss {} vs the inference forward's {infer}",
        step.loss
    );
    assert!(r_train <= 1e-4, "loss {} vs transformers {want_loss}", step.loss);
    for (name, r) in &results {
        assert!(*r <= 1e-2, "{name}: rel err {r:.3e} > 1e-2");
    }
}

/// The real 2B checkpoint, in `precision` as `load_tower` loads it (bf16 is
/// how a bf16 model trains).
fn real_2b_tower(rt: &Arc<GpuRuntime>, precision: Precision) -> Qwen35Model {
    let path = std::env::var("QWEN35_2B_SAFETENSORS").expect("set QWEN35_2B_SAFETENSORS to a Qwen3.5-2B .safetensors file");
    let st = SafeTensors::open(Path::new(&path)).unwrap();
    Qwen35Model::load_tower(rt, &st, "model.language_model.", Qwen35Config::qwen35_2b().unwrap(), precision).unwrap()
}

/// 512 natural-text ids (`tools/qwen35_ref/make_text_ids.py`).
fn text_ids() -> Vec<u32> {
    npy_f64(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_text_ids.npy"))
        .1
        .iter()
        .map(|&x| x as u32)
        .collect()
}

/// [`rel`] for every parameter of `got` against `want`, one layer (or the
/// embedding) at a time: the 2B's gradients do not fit on the host twice
/// in f64.
fn streamed_rel(cfg: &Qwen35Config, got: &Qwen35Grads, want: &Qwen35Grads) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for part in std::iter::once(usize::MAX).chain(0..got.layers.len()) {
        for ((name, a), (_, b)) in named(cfg, got, "", Some(part))
            .into_iter()
            .zip(named(cfg, want, "", Some(part)))
        {
            out.push((name, rel(&a, &b)));
        }
    }
    out
}

/// The 2B step on a bf16-stored model (bf16 weights and layer inputs, f32
/// accumulation) against tessl's exact-f32 step on the same checkpoint,
/// which `real_2b_step_matches_transformers` holds to transformers' fp32
/// autograd. Bounds written before the first run: the 2B bf16-operand
/// test's, the loss within 2^-7 and every gradient within 2^-4 of its
/// parameter's largest. The ids are natural text
/// (`tests/fixtures/qwen35_text_ids.npy`, `tools/qwen35_ref/make_text_ids.py`),
/// as the bound's own fixture is. On uniform random ids (loss 13.4) the
/// first run failed it: layer 0's `in_proj_a` 1.17e-1, `A_log` 8.3e-2,
/// `dt_bias` 7.1e-2, where bf16 operands alone, with f32 storage, already
/// gave 7.4e-2, 1.01e-1 and 4.2e-2: the bound does not hold for random ids
/// on either lane. On these ids (2026-10-08, M5 Pro) it fails on one of 320
/// parameters, kept failing rather than loosened: the loss is 7.1e-5 off and
/// 319 gradients are within 2^-4, but `layers.0.linear_attn.dt_bias` is
/// 1.04e-1 off (bf16 operands alone: 3.3e-2). Layer 0's own input is the
/// bf16 embedding, so that error arrives through the gradient from the 23
/// rounded layer boundaries above it, onto a 16-element gradient summed
/// over 512 tokens. Needs `QWEN35_2B_SAFETENSORS`.
#[test]
#[ignore]
fn real_2b_step_on_bf16_storage_stays_near_the_f32_step() {
    let rt = GpuRuntime::new().unwrap();
    let ids = text_ids();
    let f32_model = real_2b_tower(&rt, Precision::F32);
    let exact = f32_model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    let bf16_model = real_2b_tower(&rt, Precision::Bf16);
    let step = bf16_model.train_step(&ids, GemmOperands::Bf16).unwrap();
    let r_loss = (step.loss - exact.loss).abs() / exact.loss;
    eprintln!(
        "loss: bf16 storage {:.6}, f32 {:.6} (rel {r_loss:.2e})",
        step.loss, exact.loss
    );
    // Both gradient sets are f32 tensors of the same layouts. The f32
    // model's bf16-operand step is printed beside each: how much of the gap
    // operand rounding alone makes on these ids.
    let operands = f32_model.train_step(&ids, GemmOperands::Bf16).unwrap();
    let cfg = f32_model.config();
    let results = streamed_rel(cfg, &step.grads, &exact.grads);
    let op = streamed_rel(cfg, &operands.grads, &exact.grads);
    drop(operands);
    let worst = results.iter().map(|r| r.1).fold(0.0, f64::max);
    for ((name, r), (_, o)) in results.iter().zip(&op) {
        eprintln!("{name}: storage {r:.2e}, operands alone {o:.2e}");
    }
    eprintln!("worst {worst:.3e}");
    assert!(r_loss <= 2f64.powi(-7), "loss {} vs f32 {}", step.loss, exact.loss);
    for (name, r) in &results {
        assert!(*r <= 2f64.powi(-4), "{name}: rel err {r:.3e} > 2^-4");
    }
}

/// A quick loss curve (one seed; quick until it is run over several
/// seeds): the 2B trained `STEPS` AdamW steps on four fixed natural-text
/// sequences of 128 tokens in turn, at f32 and at each stored precision, every step's
/// loss printed as CSV. Bounds written before the first run: at every step
/// each stored precision's loss is within 2% of the f32 run's, and its total
/// drop over the run is within 25% of the f32 run's drop (which must be at
/// least 0.1 nats, or the curve shows nothing). Needs
/// `QWEN35_2B_SAFETENSORS`; each run holds at most ~30 GB and is dropped
/// before the next.
#[test]
#[ignore]
fn real_2b_loss_curve_tracks_the_f32_step() {
    use tessl::qwen35_adamw::{AdamW, AdamWConfig, AdamWHyper, MomentStorage, UpdateRule};
    const STEPS: usize = 30;
    let text = text_ids();
    let seqs: Vec<Vec<u32>> = text.chunks(128).map(<[u32]>::to_vec).collect();
    let hyper = AdamWHyper {
        lr: 2e-5,
        ..AdamWHyper::default()
    };
    let curve = |precision: Precision, config: AdamWConfig| -> Vec<f64> {
        let rt = GpuRuntime::new().unwrap();
        let model = real_2b_tower(&rt, precision);
        let operands = if precision == Precision::F32 {
            GemmOperands::ExactF32
        } else {
            GemmOperands::Bf16
        };
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        let mut state = AdamW::with_config(&model, config).unwrap();
        let wd = model.default_weight_decay(0.0).unwrap();
        let mut losses = Vec::with_capacity(STEPS);
        for step in 0..STEPS {
            let ids = &seqs[step % seqs.len()];
            let loss = model
                .train_step_into(ids, operands, Supervise::Causal, &bank, false)
                .unwrap();
            model.adamw_step(&bank, &mut state, &hyper, &wd).unwrap();
            losses.push(loss);
        }
        eprintln!("{}: {}", state.describe(), model.describe());
        losses
    };
    let base = curve(Precision::F32, AdamWConfig::F32);
    let configs = [
        AdamWConfig {
            update: UpdateRule::F32Master,
            moments: MomentStorage::F32,
        },
        AdamWConfig {
            update: UpdateRule::Bf16Kahan,
            moments: MomentStorage::Bf16,
        },
        AdamWConfig {
            update: UpdateRule::Bf16Kahan,
            moments: MomentStorage::Block8,
        },
        AdamWConfig {
            update: UpdateRule::Bf16Stochastic { seed: 1 },
            moments: MomentStorage::Block8,
        },
    ];
    let runs: Vec<Vec<f64>> = configs.iter().map(|&c| curve(Precision::Bf16, c)).collect();
    println!("quick (single seed, natural-text ids)");
    println!(
        "step,f32,{}",
        configs.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
    );
    for s in 0..STEPS {
        let row: Vec<String> = runs.iter().map(|r| format!("{:.5}", r[s])).collect();
        println!("{s},{:.5},{}", base[s], row.join(","));
    }
    // Over the last pass through the four sequences, against the first.
    let drop = |r: &[f64]| r[..4].iter().sum::<f64>() / 4.0 - r[STEPS - 4..].iter().sum::<f64>() / 4.0;
    let base_drop = drop(&base);
    assert!(base_drop >= 0.1, "the f32 run's loss fell only {base_drop:.4}");
    for (c, r) in configs.iter().zip(&runs) {
        for s in 0..STEPS {
            assert!(
                (r[s] - base[s]).abs() <= 0.02 * base[s],
                "{c} step {s}: loss {} vs f32 {}",
                r[s],
                base[s]
            );
        }
        let d = drop(r);
        assert!(
            (d - base_drop).abs() <= 0.25 * base_drop,
            "{c}: the loss fell {d:.4}, the f32 run's {base_drop:.4}"
        );
    }
}

/// The 2B step on bf16 GEMM operands against the same float32 transformers
/// reference as [`real_2b_step_matches_transformers`] (not transformers under
/// bf16 autocast, which rounds at other boundaries and so computes another
/// function). Bounds written before the first run: the loss within 2^-7 and
/// every compared gradient within 2^-4 of its parameter's largest, twice the
/// tiny fixture's (2^-8, 2^-5) for 24 layers instead of 2; the exact step's
/// own gap to transformers (3.9e-3) is inside them.
#[test]
#[ignore]
fn real_2b_step_on_bf16_operands_stays_near_transformers() {
    let (dir, cfg, model) = real_2b();
    let ids = ids(&dir);
    let step = model.train_step(&ids, GemmOperands::Bf16).unwrap();
    let (_, want_loss) = npy_f64(&dir.join("loss.npy"));
    let want_loss = want_loss[0];
    let r_loss = (step.loss - want_loss).abs() / want_loss;
    eprintln!(
        "loss: bf16 operands {:.8}, transformers {want_loss:.8} (rel {r_loss:.2e})",
        step.loss
    );
    let (results, worst) = real_2b_results(&dir, &cfg, &step);
    assert!(
        r_loss <= 2f64.powi(-7),
        "loss {} vs transformers {want_loss}",
        step.loss
    );
    for (name, r) in &results {
        assert!(
            *r <= 2f64.powi(-4),
            "{name}: rel err {r:.3e} > 2^-4 (worst {worst:.3e})"
        );
    }
}
