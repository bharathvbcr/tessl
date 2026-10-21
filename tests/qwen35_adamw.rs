//! `Qwen35Model::adamw_step` against torch.optim.AdamW's formula in f64.
//!
//! The reference is the single-tensor path of `torch.optim.AdamW` (amsgrad
//! and maximize off) evaluated on the host in f64, fed each step with the
//! parameters and gradients tessl itself holds (read through the parameter
//! table, the zero-centred norms as their stored `w`). Its moments are its
//! own, carried in f64 across steps.
//!
//! Bound, set before the first run: `2e-6` absolute per element, every entry
//! alike. At lr = 1e-2 an f32 update is good to ~1e-9, the parameter's own
//! rounding to ~6e-8 near 1, while a semantic error is 1e-5 or more (decay
//! after the moments: lr^2 * wd; a missing bias correction at step 1: 0.1 * lr;
//! eps outside the division by sqrt(bc2) at eps = 1e-2: ~0.3 * lr).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::qwen35_adamw::{
    excluded_from_weight_decay, AdamW, AdamWConfig, AdamWHyper, Moment, MomentStorage, UpdateRule, MOMENT_BLOCK,
};
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_params::ParamInfo;
use tessl::qwen35_train::{MixerGrads, Qwen35Grads, Supervise};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits, DType, GpuBuffer};
use tessl::{GpuRuntime, Tensor};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn load() -> (Arc<GpuRuntime>, Qwen35Model) {
    let dir = fixture();
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let model = Qwen35Model::load(&rt, &st, "model.", cfg, Precision::F32).unwrap();
    (rt, model)
}

fn ids() -> Vec<u32> {
    read_npy(&fixture().join("ids.npy"))
        .unwrap()
        .i64_slice()
        .unwrap()
        .iter()
        .map(|&x| x as u32)
        .collect()
}

/// Every parameter (`grads: false`) or the given step's gradient, on the host.
fn host(
    rt: &Arc<GpuRuntime>,
    model: &Qwen35Model,
    table: &[ParamInfo],
    grads: Option<&tessl::qwen35_train::Qwen35Grads>,
) -> Vec<Vec<f64>> {
    let ts: Vec<Tensor> = table
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect();
    match grads {
        None => model.read_parameters(&ts).unwrap(),
        Some(g) => model.read_gradients(g, &ts).unwrap(),
    }
    ts.iter()
        .map(|t| t.read_f32().unwrap().iter().map(|&x| f64::from(x)).collect())
        .collect()
}

/// `steps` steps of train_step + adamw_step from a fresh state, each checked
/// against the f64 reference; returns the worst error seen.
fn run(hyper: AdamWHyper, wd_all: f32, steps: usize) -> (f64, f64) {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(wd_all).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let ids = ids();
    let mut m: Vec<Vec<f64>> = table.iter().map(|p| vec![0.0; p.shape.iter().product()]).collect();
    let mut v = m.clone();
    let (mut mu, mut vu) = (m.clone(), m.clone());
    let mut sensitivity = 0.0f64;
    let mut worst = 0.0f64;
    for step in 1..=steps {
        let s = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
        let p0 = host(&rt, &model, &table, None);
        let g = host(&rt, &model, &table, Some(&s.grads));
        let sq: f64 = g.iter().flatten().map(|x| x * x).sum();
        let got = model.grad_sq_norm(&s.grads).unwrap();
        assert!((got - sq).abs() <= 1e-5 * sq, "step {step}: grad_sq_norm {got} vs {sq}");
        // torch scales `.grad` in place, in f32, before the step reads it.
        let scale = hyper.grad_scale as f32;
        let gs: Vec<Vec<f64>> = g
            .iter()
            .map(|t| t.iter().map(|&x| f64::from(x as f32 * scale)).collect())
            .collect();
        model.adamw_step(&s.grads, &mut state, &hyper, &wd).unwrap();
        assert_eq!(state.step_count(), step as u64);
        let p1 = host(&rt, &model, &table, None);

        let t = step as f64;
        let (bc1, bc2) = (1.0 - hyper.beta1.powf(t), 1.0 - hyper.beta2.powf(t));
        for (i, info) in table.iter().enumerate() {
            let mut moved = false;
            for k in 0..p0[i].len() {
                let decayed = p0[i][k] * (1.0 - hyper.lr * f64::from(wd[i]));
                let mut w = decayed;
                m[i][k] += (1.0 - hyper.beta1) * (gs[i][k] - m[i][k]);
                v[i][k] = v[i][k] * hyper.beta2 + (1.0 - hyper.beta2) * gs[i][k] * gs[i][k];
                let denom = v[i][k].sqrt() / bc2.sqrt() + hyper.eps;
                w -= hyper.lr / bc1 * m[i][k] / denom;
                // The same update on the unscaled gradient: how far the scale
                // moves the result, so a test can show it is not lost in the bound.
                mu[i][k] += (1.0 - hyper.beta1) * (g[i][k] - mu[i][k]);
                vu[i][k] = vu[i][k] * hyper.beta2 + (1.0 - hyper.beta2) * g[i][k] * g[i][k];
                let wu = decayed - hyper.lr / bc1 * mu[i][k] / (vu[i][k].sqrt() / bc2.sqrt() + hyper.eps);
                sensitivity = sensitivity.max((w - wu).abs());
                let err = (p1[i][k] - w).abs();
                let bound = 2e-6;
                assert!(
                    err <= bound,
                    "step {step} {}[{k}]: {} vs reference {w} (err {err:.3e})",
                    info.name,
                    p1[i][k]
                );
                worst = worst.max(err);
                moved |= p1[i][k] != p0[i][k];
            }
            assert!(moved, "step {step}: {} did not move", info.name);
        }
    }
    (worst, sensitivity)
}

#[test]
fn steps_match_torch_adamw_in_f64() {
    let (worst, _) = run(
        AdamWHyper {
            lr: 1e-2,
            ..AdamWHyper::default()
        },
        0.1,
        5,
    );
    eprintln!("eps 1e-8: worst error {worst:.2e}");
}

/// A large eps separates `sqrt(v) / sqrt(bc2) + eps` from
/// `(sqrt(v) + eps) / sqrt(bc2)` at the first steps, where `sqrt(bc2)` is ~0.03.
#[test]
fn eps_sits_outside_the_bias_correction() {
    let (worst, _) = run(
        AdamWHyper {
            lr: 1e-2,
            eps: 1e-2,
            ..AdamWHyper::default()
        },
        0.1,
        2,
    );
    eprintln!("eps 1e-2: worst error {worst:.2e}");
}

/// `grad_scale` is `clip_grad_norm_`'s coefficient applied to `.grad` in f32
/// before the step, and `grad_sq_norm` (checked inside `run` at every step)
/// is the norm it comes from. Adam cancels a gradient scale except against
/// eps, so eps is large here, and the scale must move the reference by 50x
/// the bound: a step that dropped the scale would fail the reference check.
#[test]
fn a_grad_scale_is_clip_grad_norms_coefficient() {
    let (worst, sensitivity) = run(
        AdamWHyper {
            lr: 1e-2,
            eps: 1e-2,
            grad_scale: 0.3,
            ..AdamWHyper::default()
        },
        0.1,
        2,
    );
    eprintln!("grad_scale 0.3: worst error {worst:.2e}, the scale moves the reference by {sensitivity:.2e}");
    assert!(
        sensitivity > 50.0 * 2e-6,
        "the scale moves the reference by only {sensitivity:.2e}"
    );
}

#[test]
fn the_default_decay_mask_is_transformers_trainers() {
    for name in [
        "norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.linear_attn.norm.weight",
        "layers.0.linear_attn.dt_bias",
        "layers.3.self_attn.q_norm.weight",
        "layers.3.self_attn.k_norm.weight",
    ] {
        assert!(excluded_from_weight_decay(name), "{name} should take no weight decay");
    }
    for name in [
        "embed_tokens.weight",
        "layers.0.linear_attn.A_log",
        "layers.0.linear_attn.conv1d.weight",
        "layers.0.linear_attn.in_proj_qkv.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.3.self_attn.o_proj.weight",
        "layers.3.normalizer.weight",
    ] {
        assert!(!excluded_from_weight_decay(name), "{name} should be decayed");
    }
}

#[test]
fn refusals_move_nothing() {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let s = model.train_step(&ids(), GemmOperands::ExactF32).unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let before = host(&rt, &model, &table, None);
    let ok = AdamWHyper::default();
    let e = |r: Result<(), String>, needle: &str| {
        let m = r.err().unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    e(
        model.adamw_step(&s.grads, &mut state, &ok, &wd[1..]),
        "weight decays for",
    );
    e(
        model.adamw_step(&s.grads, &mut state, &AdamWHyper { beta1: 1.0, ..ok }, &wd),
        "beta1 1 must lie in [0, 1)",
    );
    for bad in [f64::NAN, f64::INFINITY, -0.5] {
        e(
            model.adamw_step(&s.grads, &mut state, &AdamWHyper { grad_scale: bad, ..ok }, &wd),
            "must be finite and >= 0",
        );
    }
    e(
        model.adamw_step(&s.grads, &mut state, &AdamWHyper { beta2: -0.1, ..ok }, &wd),
        "beta2 -0.1",
    );
    e(
        model.adamw_step(&s.grads, &mut state, &AdamWHyper { eps: 0.0, ..ok }, &wd),
        "eps 0",
    );
    e(
        model.adamw_step(&s.grads, &mut state, &AdamWHyper { lr: f64::NAN, ..ok }, &wd),
        "lr NaN",
    );
    let mut bad = wd.clone();
    bad[3] = -1.0;
    e(
        model.adamw_step(&s.grads, &mut state, &ok, &bad),
        "weight decay -1 must be finite",
    );
    assert_eq!(state.step_count(), 0);
    assert_eq!(
        host(&rt, &model, &table, None),
        before,
        "a refused step moved the parameters"
    );
}

/// Gradients or moments on another runtime (a bank built field by field, or
/// an `AdamW` made for a model loaded there) are refused by `adamw_step`
/// and `grad_sq_norm` — the two callers of the window check — and by the
/// gradient-layout copies, before anything moves.
#[test]
fn adamw_and_grad_sq_norm_refuse_a_foreign_runtime() {
    let (rt, model) = load();
    let (_, elsewhere) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let ok = AdamWHyper::default();
    let s = model.train_step(&ids(), GemmOperands::ExactF32).unwrap();
    let before = host(&rt, &model, &table, None);
    let e = |r: Result<(), String>, what: &str| {
        let m = r.err().unwrap_or_else(|| panic!("{what}: accepted"));
        assert!(m.contains("belongs to another runtime"), "{what}: {m}");
    };

    let mut foreign = elsewhere.train_step(&ids(), GemmOperands::ExactF32).unwrap().grads;
    // The first slot is this model's, so the check must reach past it.
    foreign.embed = s.grads.embed.clone();
    let mut state = AdamW::new(&model).unwrap();
    let _ = rt.take_dispatch_count();
    e(model.adamw_step(&foreign, &mut state, &ok, &wd), "adamw_step gradients");
    e(model.grad_sq_norm(&foreign).map(|_| ()), "grad_sq_norm");
    let dst: Vec<Tensor> = table
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect();
    e(model.read_gradients(&foreign, &dst), "read_gradients");
    assert_eq!(rt.take_dispatch_count(), 0, "a refusal encoded work");

    let mut their_state = AdamW::new(&elsewhere).unwrap();
    e(
        model.adamw_step(&s.grads, &mut their_state, &ok, &wd),
        "adamw_step moments",
    );
    e(
        model.write_adamw_moment(&mut their_state, Moment::First, &dst),
        "write_adamw_moment",
    );
    assert_eq!(rt.take_dispatch_count(), 0, "a refusal encoded work");
    assert_eq!(state.step_count(), 0);
    assert_eq!(their_state.step_count(), 0);
    assert_eq!(
        host(&rt, &model, &table, None),
        before,
        "a refused step moved the parameters"
    );
}

/// A checkpoint of the parameters, both moments and the step count, restored
/// into a freshly loaded model and fresh state, resumes the run exactly: the
/// third step gives the uninterrupted run's parameters bit for bit. Without
/// the moments or the count the resumed step differs (checked too).
#[test]
fn a_checkpointed_run_resumes_bit_for_bit() {
    let ids = ids();
    let hyper = AdamWHyper {
        lr: 1e-2,
        ..AdamWHyper::default()
    };
    let (rt, a) = load();
    let table = a.parameter_table().unwrap();
    let wd = a.default_weight_decay(0.1).unwrap();
    let alloc = |rt: &Arc<GpuRuntime>| -> Vec<Tensor> {
        table
            .iter()
            .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
            .collect()
    };
    let mut sa = AdamW::new(&a).unwrap();
    for _ in 0..2 {
        let s = a.train_step(&ids, GemmOperands::ExactF32).unwrap();
        a.adamw_step(&s.grads, &mut sa, &hyper, &wd).unwrap();
    }
    let (p2, m2, v2) = (alloc(&rt), alloc(&rt), alloc(&rt));
    a.read_parameters(&p2).unwrap();
    a.read_adamw_moment(&sa, Moment::First, &m2).unwrap();
    a.read_adamw_moment(&sa, Moment::Second, &v2).unwrap();
    let s = a.train_step(&ids, GemmOperands::ExactF32).unwrap();
    a.adamw_step(&s.grads, &mut sa, &hyper, &wd).unwrap();
    let want = host(&rt, &a, &table, None);

    // resume(moments, count): a fresh model and state from the checkpoint.
    let resume = |moments: bool, count: bool| -> Vec<Vec<f64>> {
        let (rb, b) = load();
        let copy = |src: &[Tensor]| -> Vec<Tensor> {
            src.iter()
                .map(|t| {
                    let n = rb.alloc_tensor_f32(t.shape()).unwrap();
                    n.write_f32(&t.read_f32().unwrap()).unwrap();
                    n
                })
                .collect()
        };
        b.write_parameters(&copy(&p2)).unwrap();
        let mut sb = AdamW::new(&b).unwrap();
        if moments {
            b.write_adamw_moment(&mut sb, Moment::First, &copy(&m2)).unwrap();
            b.write_adamw_moment(&mut sb, Moment::Second, &copy(&v2)).unwrap();
            let back = alloc(&rb);
            b.read_adamw_moment(&sb, Moment::Second, &back).unwrap();
            for (x, y) in back.iter().zip(&v2) {
                assert_eq!(
                    x.read_f32().unwrap(),
                    y.read_f32().unwrap(),
                    "a moment did not round-trip"
                );
            }
        }
        if count {
            sb.set_step_count(2);
        }
        let s = b.train_step(&ids, GemmOperands::ExactF32).unwrap();
        b.adamw_step(&s.grads, &mut sb, &hyper, &wd).unwrap();
        assert_eq!(sb.step_count(), if count { 3 } else { 1 });
        host(&rb, &b, &table, None)
    };
    let bits =
        |v: &[Vec<f64>]| -> Vec<Vec<u64>> { v.iter().map(|t| t.iter().map(|x| x.to_bits()).collect()).collect() };
    assert_eq!(
        bits(&resume(true, true)),
        bits(&want),
        "the resumed step is not the uninterrupted one"
    );
    assert_ne!(bits(&resume(false, true)), bits(&want), "moments made no difference");
    assert_ne!(
        bits(&resume(true, false)),
        bits(&want),
        "the step count made no difference"
    );

    // A misshapen checkpoint is refused before anything is written.
    let mut sc = AdamW::new(&a).unwrap();
    let e = a.write_adamw_moment(&mut sc, Moment::First, &m2[1..]).unwrap_err();
    assert!(e.contains("tensors for"), "{e}");
}

/// One zero-gradient AdamW step over the tiny fixture, fingerprinted before the
/// generic-kernel refactor. A later edit that retunes the scalar formation or
/// the dispatch changes this hash.
#[test]
fn qwen35_adamw_zero_grad_step_keeps_its_bits() {
    let (rt, model) = load();
    let mut state = AdamW::new(&model).unwrap();
    let grads = tessl::qwen35_train::Qwen35Grads::zeros_like(&model).unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let hyper = AdamWHyper {
        lr: 1e-2,
        ..AdamWHyper::default()
    };
    model.adamw_step(&grads, &mut state, &hyper, &wd).unwrap();
    let table = model.parameter_table().unwrap();
    let got = host(&rt, &model, &table, None);
    let mut hash: u64 = 0;
    let mut n = 0usize;
    for tensor in &got {
        for x in tensor {
            hash = hash.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(x.to_bits());
            n += 1;
        }
    }
    // Filled from the pre-refactor run. A mismatch names the actual hash.
    const WANT: u64 = 0xb94c9db440000000;
    assert_eq!(
        (hash, n),
        (WANT, n),
        "zero-grad adamw fingerprint is {hash:#x} over {n} elements"
    );
}

/// The step counter is a `u64`. One past `u64::MAX` must be an error, and the
/// counter must stay at `u64::MAX`. In a release build without overflow checks
/// the pre-fix `step + 1` wraps to 0, the bias correction becomes `1 - beta^0
/// = 0`, and the parameter update divides by that.
#[test]
fn adamw_step_at_u64_max_does_not_wrap_to_zero() {
    let (_rt, model) = load();
    let mut state = AdamW::new(&model).unwrap();
    state.set_step_count(u64::MAX);
    let grads = tessl::qwen35_train::Qwen35Grads::zeros_like(&model).unwrap();
    let wd = model.default_weight_decay(0.0).unwrap();
    let err = model
        .adamw_step(&grads, &mut state, &AdamWHyper::default(), &wd)
        .expect_err("u64::MAX + 1 must not become step 0");
    assert!(
        err.contains("u64") || err.contains("overflow"),
        "unexpected refusal: {err}"
    );
    assert_eq!(state.step_count(), u64::MAX);
}

// ---- stored precision ------------------------------------------------------
//
// The fixture as a bf16 tower (its weights are bf16 on disk, so the values are
// the f32 model's), stepped on bf16 GEMM operands. Each test checks one step
// at a time against the f64 reference started from tessl's own state before
// that step, so a bound is one step's rounding, not a drifted trajectory.

fn load_bf16() -> (Arc<GpuRuntime>, Qwen35Model) {
    let dir = fixture();
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let model = Qwen35Model::load_tower(&rt, &st, "model.", cfg, Precision::Bf16).unwrap();
    (rt, model)
}

fn alloc_table(rt: &Arc<GpuRuntime>, table: &[ParamInfo]) -> Vec<Tensor> {
    table
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect()
}

fn to_host(ts: &[Tensor]) -> Vec<Vec<f64>> {
    ts.iter()
        .map(|t| t.read_f32().unwrap().iter().map(|&x| f64::from(x)).collect())
        .collect()
}

fn moments(
    rt: &Arc<GpuRuntime>,
    model: &Qwen35Model,
    table: &[ParamInfo],
    s: &AdamW,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let (m, v) = (alloc_table(rt, table), alloc_table(rt, table));
    model.read_adamw_moment(s, Moment::First, &m).unwrap();
    model.read_adamw_moment(s, Moment::Second, &v).unwrap();
    (to_host(&m), to_host(&v))
}

fn aux(rt: &Arc<GpuRuntime>, model: &Qwen35Model, table: &[ParamInfo], s: &AdamW) -> Vec<Vec<f64>> {
    let a = alloc_table(rt, table);
    model.read_adamw_aux(s, &a).unwrap();
    to_host(&a)
}

/// torch's AdamW on one element in f64 from `w` and the moments `m`, `v`
/// (updated in place): the new value.
fn reference(h: &AdamWHyper, step: u64, wd: f32, w: f64, g: f64, m: &mut f64, v: &mut f64) -> f64 {
    let t = step as f64;
    let (bc1, bc2) = (1.0 - h.beta1.powf(t), 1.0 - h.beta2.powf(t));
    let g = f64::from(g as f32 * h.grad_scale as f32);
    *m += (1.0 - h.beta1) * (g - *m);
    *v = *v * h.beta2 + (1.0 - h.beta2) * g * g;
    w * (1.0 - h.lr * f64::from(wd)) - h.lr / bc1 * *m / (v.sqrt() / bc2.sqrt() + h.eps)
}

fn bf16(x: f64) -> f64 {
    f64::from(tessl::tensor::bf16_bits_to_f32(tessl::tensor::f32_to_bf16_bits(
        x as f32,
    )))
}

/// One bf16 ulp at `x` (the spacing of bf16 values around it).
fn bf16_ulp(x: f64) -> f64 {
    let e = (x.abs().max(f64::from(f32::MIN_POSITIVE))).log2().floor();
    2f64.powf(e - 7.0)
}

const LR: f64 = 1e-2;

fn hyper() -> AdamWHyper {
    AdamWHyper {
        lr: LR,
        ..AdamWHyper::default()
    }
}

/// [`UpdateRule::F32Master`] with f32 moments: the master takes torch's
/// update (the f32 bound of `run`, 2e-6, set before the first run), every
/// bf16 weight is its master rounded to nearest bit for bit, and the f32
/// entries are their masters. With bf16 moments, each moment reads back as a
/// bf16 value, and the master's error to the f64 reference (started from
/// tessl's widened moments) is within 2e-6 + lr * 2^-7: the stored moments
/// are the reference's own start, so only the new moments' single rounding
/// (2^-9 relative each, under 2^-8 in `m / sqrt(v)`) moves the update.
#[test]
fn an_f32_master_takes_torchs_update_and_rounds_the_weights() {
    for moments_kind in [MomentStorage::F32, MomentStorage::Bf16] {
        let (rt, model) = load_bf16();
        let table = model.parameter_table().unwrap();
        let wd = model.default_weight_decay(0.1).unwrap();
        let config = AdamWConfig {
            update: UpdateRule::F32Master,
            moments: moments_kind,
        };
        let mut state = AdamW::with_config(&model, config).unwrap();
        assert_eq!(state.config(), config);
        assert!(state.describe().contains("update=f32-master"), "{}", state.describe());
        let bound = match moments_kind {
            MomentStorage::F32 => 2e-6,
            _ => 2e-6 + LR * 2f64.powi(-7),
        };
        let mut worst = 0.0f64;
        for step in 1..=4u64 {
            let s = model.train_step(&ids(), GemmOperands::Bf16).unwrap();
            let g = host(&rt, &model, &table, Some(&s.grads));
            let master0 = aux(&rt, &model, &table, &state);
            let (mut m, mut v) = moments(&rt, &model, &table, &state);
            model.adamw_step(&s.grads, &mut state, &hyper(), &wd).unwrap();
            let master1 = aux(&rt, &model, &table, &state);
            let p1 = host(&rt, &model, &table, None);
            let (m1, v1) = moments(&rt, &model, &table, &state);
            for (i, info) in table.iter().enumerate() {
                for k in 0..g[i].len() {
                    let want = reference(
                        &hyper(),
                        step,
                        wd[i],
                        master0[i][k],
                        g[i][k],
                        &mut m[i][k],
                        &mut v[i][k],
                    );
                    let err = (master1[i][k] - want).abs();
                    worst = worst.max(err);
                    assert!(
                        err <= bound,
                        "{moments_kind:?} step {step} {}[{k}]: master {} vs {want} (err {err:.3e})",
                        info.name,
                        master1[i][k]
                    );
                    assert_eq!(
                        p1[i][k].to_bits(),
                        bf16_or_f32(&info.name, master1[i][k]).to_bits(),
                        "{} [{k}]: the weight is not its master rounded to nearest",
                        info.name
                    );
                    if moments_kind == MomentStorage::Bf16 {
                        assert_eq!(m1[i][k], bf16(m1[i][k]), "{}: a bf16 moment is not bf16", info.name);
                        assert_eq!(v1[i][k], bf16(v1[i][k]), "{}: a bf16 moment is not bf16", info.name);
                    }
                }
            }
        }
        eprintln!("{config}: worst master error {worst:.2e} (bound {bound:.2e})");
    }
}

/// The fixture's bf16-stored entries are the matrices; the rest are f32.
fn stored_bf16(name: &str) -> bool {
    name.ends_with("proj.weight") || name == "embed_tokens.weight" || name.contains("proj_")
}

fn bf16_or_f32(name: &str, x: f64) -> f64 {
    if stored_bf16(name) {
        bf16(x)
    } else {
        x
    }
}

/// [`UpdateRule::Bf16Kahan`]: one step from the effective value `p + c`
/// lands on torch's update of it, to within one rounding of the new
/// compensation (half a bf16 ulp of `c`, which is itself under half an ulp
/// of `p`) plus the f32 bound:
/// `|p1 + c1 - ref| <= 2^-9 |c1| + 2^-17 |ref| + 2e-6`. Set before the first
/// run. At lr = 1e-4 most updates are below
/// half an ulp of their weight, which plain rounding to nearest would drop:
/// after the steps the bf16 weights with their compensation are much nearer
/// the uninterrupted reference than the bf16 weights alone (checked, 4x).
#[test]
fn kahan_compensation_keeps_what_bf16_rounding_drops() {
    let (rt, model) = load_bf16();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(0.0).unwrap();
    let mut state = AdamW::with_config(
        &model,
        AdamWConfig {
            update: UpdateRule::Bf16Kahan,
            moments: MomentStorage::F32,
        },
    )
    .unwrap();
    let h = AdamWHyper {
        lr: 1e-4,
        ..AdamWHyper::default()
    };
    // The uninterrupted reference, in f64 from the loaded weights.
    let mut ref_w = host(&rt, &model, &table, None);
    let (mut rm, mut rv) = moments(&rt, &model, &table, &state);
    let mut worst = 0.0f64;
    for step in 1..=8u64 {
        let s = model.train_step(&ids(), GemmOperands::Bf16).unwrap();
        let g = host(&rt, &model, &table, Some(&s.grads));
        let p0 = host(&rt, &model, &table, None);
        let c0 = aux(&rt, &model, &table, &state);
        let (mut m, mut v) = moments(&rt, &model, &table, &state);
        model.adamw_step(&s.grads, &mut state, &h, &wd).unwrap();
        let p1 = host(&rt, &model, &table, None);
        let c1 = aux(&rt, &model, &table, &state);
        for (i, info) in table.iter().enumerate() {
            for k in 0..g[i].len() {
                let want = reference(
                    &h,
                    step,
                    wd[i],
                    p0[i][k] + c0[i][k],
                    g[i][k],
                    &mut m[i][k],
                    &mut v[i][k],
                );
                let got = p1[i][k] + c1[i][k];
                let err = (got - want).abs();
                let bound = 2f64.powi(-9) * c1[i][k].abs() + 2f64.powi(-17) * want.abs() + 2e-6;
                worst = worst.max(err);
                assert!(
                    err <= bound,
                    "step {step} {}[{k}]: p + c = {got} vs {want} (err {err:.3e} > {bound:.3e})",
                    info.name
                );
                ref_w[i][k] = reference(&h, step, wd[i], ref_w[i][k], g[i][k], &mut rm[i][k], &mut rv[i][k]);
            }
        }
    }
    let p = host(&rt, &model, &table, None);
    let c = aux(&rt, &model, &table, &state);
    let (mut with_c, mut without) = (0.0f64, 0.0f64);
    for (i, info) in table.iter().enumerate() {
        if !stored_bf16(&info.name) {
            continue;
        }
        for k in 0..p[i].len() {
            with_c += (p[i][k] + c[i][k] - ref_w[i][k]).abs();
            without += (p[i][k] - ref_w[i][k]).abs();
        }
    }
    eprintln!(
        "kahan: worst one-step error {worst:.2e}; drift from the reference with c {with_c:.3e}, without {without:.3e}"
    );
    assert!(
        without > 4.0 * with_c,
        "the compensation does not hold what rounding dropped: {with_c:.3e} with, {without:.3e} without"
    );
}

/// [`UpdateRule::Bf16Stochastic`]: every bf16 weight after a step is one of
/// the two bf16 values around torch's update of it (within one ulp), the
/// rounding is unbiased (over the embedding's elements, the mean of
/// `(p1 - ref) / ulp` is within 4 standard errors of 0, each term's spread
/// being at most 1/2), the same seed gives the same bits, and another seed
/// different ones. Bounds set before the first run.
#[test]
fn stochastic_rounding_is_unbiased_and_reproducible() {
    let run = |seed: u64| -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<ParamInfo>) {
        let (rt, model) = load_bf16();
        let table = model.parameter_table().unwrap();
        let wd = model.default_weight_decay(0.1).unwrap();
        let mut state = AdamW::with_config(
            &model,
            AdamWConfig {
                update: UpdateRule::Bf16Stochastic { seed },
                moments: MomentStorage::F32,
            },
        )
        .unwrap();
        assert!(
            state.describe().contains(&format!("seed={seed}")),
            "{}",
            state.describe()
        );
        let s = model.train_step(&ids(), GemmOperands::Bf16).unwrap();
        let g = host(&rt, &model, &table, Some(&s.grads));
        let p0 = host(&rt, &model, &table, None);
        model.adamw_step(&s.grads, &mut state, &hyper(), &wd).unwrap();
        let p1 = host(&rt, &model, &table, None);
        let mut want = Vec::new();
        for (i, _) in table.iter().enumerate() {
            want.push(
                (0..g[i].len())
                    .map(|k| reference(&hyper(), 1, wd[i], p0[i][k], g[i][k], &mut 0.0, &mut 0.0))
                    .collect::<Vec<f64>>(),
            );
        }
        (p1, want, table)
    };
    let (p1, want, table) = run(7);
    let (mut sum, mut n) = (0.0f64, 0usize);
    for (i, info) in table.iter().enumerate() {
        for k in 0..p1[i].len() {
            let (got, w) = (p1[i][k], want[i][k]);
            if stored_bf16(&info.name) {
                let ulp = bf16_ulp(w);
                assert!(
                    (got - w).abs() <= ulp * (1.0 + 1e-3) + 2e-6,
                    "{}[{k}]: {got} is not a bf16 neighbour of {w}",
                    info.name
                );
                if info.name == "embed_tokens.weight" {
                    sum += (got - w) / ulp;
                    n += 1;
                }
            } else {
                assert!((got - w).abs() <= 2e-6, "{}[{k}]: f32 entry {got} vs {w}", info.name);
            }
        }
    }
    let mean = sum / n as f64;
    let limit = 4.0 * 0.5 / (n as f64).sqrt();
    eprintln!("stochastic rounding: mean (p - ref) / ulp over {n} elements {mean:.4} (limit {limit:.4})");
    assert!(mean.abs() <= limit, "biased: mean {mean} over {n}, limit {limit}");
    let bits = |v: &[Vec<f64>]| -> Vec<u64> { v.iter().flatten().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&run(7).0), bits(&p1), "the same seed gave different bits");
    assert_ne!(bits(&run(8).0), bits(&p1), "another seed gave the same bits");
}

/// 8-bit block moments: after each step the decoded moments are torch's
/// moments (from tessl's decoded moments before the step) to within half a
/// code step of their block: `|m - ref| <= 0.0080 max_block |ref m|` and
/// `|sqrt v - sqrt ref| <= 0.0040 max_block sqrt(ref v)`, plus 1e-12 (the
/// companded codes' widest step is at the top, `(2q + 1) / (2 * 127^2)` and
/// `/ (2 * 255^2)` of the scale). A non-zero v never decodes to zero. A
/// read written back reads back the same bits. Bounds set before the first
/// run.
#[test]
fn block8_moments_stay_within_half_a_code_of_torchs() {
    for update in [UpdateRule::F32Master, UpdateRule::Bf16Kahan] {
        let (rt, model) = load_bf16();
        let table = model.parameter_table().unwrap();
        let wd = model.default_weight_decay(0.1).unwrap();
        let mut state = AdamW::with_config(
            &model,
            AdamWConfig {
                update,
                moments: MomentStorage::Block8,
            },
        )
        .unwrap();
        assert!(state.describe().contains("moments=block8/256"), "{}", state.describe());
        for step in 1..=3u64 {
            let s = model.train_step(&ids(), GemmOperands::Bf16).unwrap();
            let g = host(&rt, &model, &table, Some(&s.grads));
            let (mut m, mut v) = moments(&rt, &model, &table, &state);
            model.adamw_step(&s.grads, &mut state, &hyper(), &wd).unwrap();
            let (m1, v1) = moments(&rt, &model, &table, &state);
            for (i, info) in table.iter().enumerate() {
                for k in 0..g[i].len() {
                    reference(&hyper(), step, wd[i], 0.0, g[i][k], &mut m[i][k], &mut v[i][k]);
                }
                for (b, chunk) in (0..g[i].len()).collect::<Vec<_>>().chunks(MOMENT_BLOCK).enumerate() {
                    let sm = chunk.iter().map(|&k| m[i][k].abs()).fold(0.0, f64::max);
                    let sv = chunk.iter().map(|&k| v[i][k].sqrt()).fold(0.0, f64::max);
                    for &k in chunk {
                        let em = (m1[i][k] - m[i][k]).abs();
                        let ev = (v1[i][k].sqrt() - v[i][k].sqrt()).abs();
                        assert!(
                            em <= 0.0080 * sm + 1e-12,
                            "{update:?} step {step} {} block {b} [{k}]: m {} vs {} (scale {sm:.3e})",
                            info.name,
                            m1[i][k],
                            m[i][k]
                        );
                        assert!(
                            ev <= 0.0040 * sv + 1e-12,
                            "{update:?} step {step} {} block {b} [{k}]: v {} vs {} (scale {sv:.3e})",
                            info.name,
                            v1[i][k],
                            v[i][k]
                        );
                        assert!(
                            v[i][k] == 0.0 || v1[i][k] > 0.0,
                            "{} [{k}]: a non-zero v decoded to zero",
                            info.name
                        );
                    }
                }
            }
        }
        // Idempotent: what a read gives, written back, reads back the same.
        let (m, v) = (alloc_table(&rt, &table), alloc_table(&rt, &table));
        model.read_adamw_moment(&state, Moment::First, &m).unwrap();
        model.read_adamw_moment(&state, Moment::Second, &v).unwrap();
        model.write_adamw_moment(&mut state, Moment::First, &m).unwrap();
        model.write_adamw_moment(&mut state, Moment::Second, &v).unwrap();
        let (m2, v2) = moments(&rt, &model, &table, &state);
        assert_eq!(to_host(&m), m2, "{update:?}: a first moment did not round-trip");
        assert_eq!(to_host(&v), v2, "{update:?}: a second moment did not round-trip");
    }
}

/// Every configuration resumes from a checkpoint (parameters, both moments,
/// any auxiliary state, the step count) bit for bit: the third step of a
/// fresh model and state restored after two steps is the uninterrupted
/// run's third step.
#[test]
fn every_stored_precision_resumes_bit_for_bit() {
    let configs = [
        AdamWConfig {
            update: UpdateRule::F32Master,
            moments: MomentStorage::Bf16,
        },
        AdamWConfig {
            update: UpdateRule::Bf16Kahan,
            moments: MomentStorage::Block8,
        },
        AdamWConfig {
            update: UpdateRule::Bf16Stochastic { seed: 11 },
            moments: MomentStorage::Block8,
        },
        AdamWConfig {
            update: UpdateRule::Bf16Stochastic { seed: 11 },
            moments: MomentStorage::F32,
        },
    ];
    let ids = ids();
    for config in configs {
        let (rt, a) = load_bf16();
        let table = a.parameter_table().unwrap();
        let wd = a.default_weight_decay(0.1).unwrap();
        let mut sa = AdamW::with_config(&a, config).unwrap();
        for _ in 0..2 {
            let s = a.train_step(&ids, GemmOperands::Bf16).unwrap();
            a.adamw_step(&s.grads, &mut sa, &hyper(), &wd).unwrap();
        }
        let (p2, m2, v2, x2) = (
            alloc_table(&rt, &table),
            alloc_table(&rt, &table),
            alloc_table(&rt, &table),
            alloc_table(&rt, &table),
        );
        a.read_parameters(&p2).unwrap();
        a.read_adamw_moment(&sa, Moment::First, &m2).unwrap();
        a.read_adamw_moment(&sa, Moment::Second, &v2).unwrap();
        let has_aux = a.read_adamw_aux(&sa, &x2).is_ok();
        let s = a.train_step(&ids, GemmOperands::Bf16).unwrap();
        a.adamw_step(&s.grads, &mut sa, &hyper(), &wd).unwrap();
        let want = host(&rt, &a, &table, None);

        let (rb, b) = load_bf16();
        let copy = |src: &[Tensor]| -> Vec<Tensor> {
            src.iter()
                .map(|t| {
                    let n = rb.alloc_tensor_f32(t.shape()).unwrap();
                    n.write_f32(&t.read_f32().unwrap()).unwrap();
                    n
                })
                .collect()
        };
        b.write_parameters(&copy(&p2)).unwrap();
        let mut sb = AdamW::with_config(&b, config).unwrap();
        b.write_adamw_moment(&mut sb, Moment::First, &copy(&m2)).unwrap();
        b.write_adamw_moment(&mut sb, Moment::Second, &copy(&v2)).unwrap();
        if has_aux {
            b.write_adamw_aux(&mut sb, &copy(&x2)).unwrap();
        }
        sb.set_step_count(2);
        let s = b.train_step(&ids, GemmOperands::Bf16).unwrap();
        b.adamw_step(&s.grads, &mut sb, &hyper(), &wd).unwrap();
        let got = host(&rb, &b, &table, None);
        assert!(got == want, "{config}: the resumed step is not the uninterrupted one");
    }
}

/// What a configuration does not cover is refused before any state exists or
/// anything moves.
#[test]
fn stored_precision_refusals() {
    let (_, f32_model) = load();
    let (rt, bf16_model) = load_bf16();
    let e = |r: Result<AdamW, String>, needle: &str| {
        let m = r.err().unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    e(AdamW::new(&bf16_model), "AdamW::with_config");
    e(AdamW::with_config(&bf16_model, AdamWConfig::F32), "does not apply");
    e(
        AdamW::with_config(
            &f32_model,
            AdamWConfig {
                update: UpdateRule::Bf16Kahan,
                moments: MomentStorage::Bf16,
            },
        ),
        "does not apply",
    );
    // An f32 model takes any moment storage.
    AdamW::with_config(
        &f32_model,
        AdamWConfig {
            update: UpdateRule::F32,
            moments: MomentStorage::Block8,
        },
    )
    .unwrap();
    let mut sr = AdamW::with_config(
        &bf16_model,
        AdamWConfig {
            update: UpdateRule::Bf16Stochastic { seed: 1 },
            moments: MomentStorage::F32,
        },
    )
    .unwrap();
    let table = bf16_model.parameter_table().unwrap();
    let err = bf16_model.read_adamw_aux(&sr, &alloc_table(&rt, &table)).unwrap_err();
    assert!(err.contains("keeps no auxiliary state"), "{err}");
    // State made for another model is refused.
    let g = tessl::qwen35_train::Qwen35Grads::zeros_like(&f32_model).unwrap();
    let wd = bf16_model.default_weight_decay(0.0).unwrap();
    let err = f32_model
        .adamw_step(&g, &mut sr, &AdamWHyper::default(), &wd)
        .unwrap_err();
    assert!(err.contains("another model"), "{err}");
    assert_eq!(sr.step_count(), 0);
}

// ---- per-parameter learning rate ---------------------------------------------

/// `adamw_step_scaled` is torch's AdamW with one param group per entry, entry
/// `i`'s lr being `lr * lr_scale[i]`: both the decoupled decay and the step
/// size take the scaled lr. Here the embedding runs at half the lr, one entry
/// at 0 (frozen, as a torch group at lr 0: the parameter keeps its bits while
/// its moments still update) and the rest at the full lr, over two steps. Each
/// step starts the f64 reference from tessl's moments before it, so a bound is
/// one step's rounding: every entry against its own scaled reference at
/// `run`'s bound (2e-6), and the frozen entry's moments within 1e-6 of their
/// operands' magnitude (`|m0| + |g|`, `v0 + g^2`; f32 is ~6e-8). The
/// embedding's half-lr update must differ from a full-lr one by far more than
/// the bound, so a dropped scale cannot pass.
#[test]
fn a_per_entry_lr_scale_is_a_torch_param_group() {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let embed = table.iter().position(|p| p.name == "embed_tokens.weight").unwrap();
    let frozen = table
        .iter()
        .position(|p| p.name.ends_with("mlp.down_proj.weight"))
        .unwrap();
    let mut scale = vec![1.0f64; table.len()];
    scale[embed] = 0.5;
    scale[frozen] = 0.0;
    let mut state = AdamW::new(&model).unwrap();
    let (mut worst, mut sensitivity) = (0.0f64, 0.0f64);
    for step in 1..=2u64 {
        let s = model.train_step(&ids(), GemmOperands::ExactF32).unwrap();
        let p0 = host(&rt, &model, &table, None);
        let g = host(&rt, &model, &table, Some(&s.grads));
        let (m0, v0) = moments(&rt, &model, &table, &state);
        let (mut m, mut v) = (m0.clone(), v0.clone());
        model
            .adamw_step_scaled(&s.grads, &mut state, &hyper(), &wd, &scale)
            .unwrap();
        assert_eq!(state.step_count(), step);
        let p1 = host(&rt, &model, &table, None);
        let (m1, v1) = moments(&rt, &model, &table, &state);
        for (i, info) in table.iter().enumerate() {
            let h = AdamWHyper {
                lr: LR * scale[i],
                ..hyper()
            };
            for k in 0..p0[i].len() {
                let (mut mf, mut vf) = (m[i][k], v[i][k]);
                let full = reference(&hyper(), step, wd[i], p0[i][k], g[i][k], &mut mf, &mut vf);
                let want = reference(&h, step, wd[i], p0[i][k], g[i][k], &mut m[i][k], &mut v[i][k]);
                let err = (p1[i][k] - want).abs();
                worst = worst.max(err);
                assert!(
                    err <= 2e-6,
                    "step {step} {}[{k}] (scale {}): {} vs {want} (err {err:.3e})",
                    info.name,
                    scale[i],
                    p1[i][k]
                );
                if i == embed {
                    sensitivity = sensitivity.max((want - full).abs());
                }
                if i == frozen {
                    assert_eq!(
                        p1[i][k].to_bits(),
                        p0[i][k].to_bits(),
                        "step {step} {}[{k}]: a scale-0 entry moved",
                        info.name
                    );
                    let gk = g[i][k];
                    for (got, want, size, what) in [
                        (m1[i][k], m[i][k], m0[i][k].abs() + gk.abs(), "m"),
                        (v1[i][k], v[i][k], v0[i][k] + gk * gk, "v"),
                    ] {
                        assert!(
                            (got - want).abs() <= 1e-6 * size + 1e-18,
                            "step {step} {}[{k}]: frozen {what} {got} vs {want}",
                            info.name
                        );
                    }
                }
            }
        }
        assert!(
            m1[frozen].iter().any(|&x| x != 0.0),
            "step {step}: the frozen entry's first moment did not update"
        );
    }
    eprintln!("lr_scale: worst error {worst:.2e}; the embedding's 0.5 moves it by {sensitivity:.2e}");
    assert!(
        sensitivity > 50.0 * 2e-6,
        "the embedding's scale moves the reference by only {sensitivity:.2e}"
    );
}

/// A wrong-length, negative, NaN or infinite `lr_scale`, or one whose scaled
/// lr overflows, is refused before anything moves: no parameter, moment or
/// step count changes, and no work is encoded.
#[test]
fn lr_scale_refusals_move_nothing() {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let s = model.train_step(&ids(), GemmOperands::ExactF32).unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let before = host(&rt, &model, &table, None);
    let (m0, v0) = moments(&rt, &model, &table, &state);
    let ones = vec![1.0f64; table.len()];
    let mut e = |hyper: AdamWHyper, scale: &[f64], needle: &str| {
        let _ = rt.take_dispatch_count();
        let r = model.adamw_step_scaled(&s.grads, &mut state, &hyper, &wd, scale);
        let m = r.err().unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        assert_eq!(rt.take_dispatch_count(), 0, "{needle}: a refusal encoded work");
    };
    e(hyper(), &ones[1..], "lr scales for");
    e(hyper(), &[ones.as_slice(), &[1.0]].concat(), "lr scales for");
    for (bad, needle) in [
        (-0.5, "lr scale -0.5 must be finite and >= 0"),
        (f64::NAN, "lr scale NaN must be finite and >= 0"),
        (f64::INFINITY, "lr scale inf must be finite and >= 0"),
    ] {
        let mut scale = ones.clone();
        scale[3] = bad;
        e(hyper(), &scale, needle);
    }
    let mut huge = ones.clone();
    huge[2] = f64::MAX;
    e(AdamWHyper { lr: 2.0, ..hyper() }, &huge, "lr inf must be finite");
    assert_eq!(state.step_count(), 0);
    assert_eq!(
        host(&rt, &model, &table, None),
        before,
        "a refused step moved the parameters"
    );
    assert_eq!(
        moments(&rt, &model, &table, &state),
        (m0, v0),
        "a refused step moved the moments"
    );
}

// ---- grad_sq_norm ----------------------------------------------------------

/// Every buffer of a bank: the buffer, the byte offset of its elements,
/// their count and type.
type Part<'a> = (&'a GpuBuffer, usize, usize, DType);

fn bank_parts(g: &Qwen35Grads) -> Vec<Part<'_>> {
    fn t(x: &Tensor) -> Part<'_> {
        (&x.buffer, x.byte_offset(), x.numel(), x.dtype)
    }
    fn b(x: &GpuBuffer) -> Part<'_> {
        (x, 0, x.nbytes() / 4, DType::F32)
    }
    let mut v = vec![t(&g.embed), b(&g.final_norm)];
    for l in &g.layers {
        v.extend([b(&l.input_norm), b(&l.post_norm)]);
        match &l.mixer {
            MixerGrads::Gdn(m) => v.extend([
                t(&m.w_in),
                t(&m.w_out),
                b(&m.conv_w),
                b(&m.a_log),
                b(&m.dt_bias),
                b(&m.norm_w),
            ]),
            MixerGrads::Attn(m) => v.extend([t(&m.w_in), t(&m.w_out), b(&m.q_norm), b(&m.k_norm)]),
        }
        v.extend([t(&l.gate), t(&l.up), t(&l.down)]);
    }
    v
}

/// Write `f(part, element)` into every element of every bank buffer (each
/// value must be exact in the buffer's dtype).
fn fill_bank(g: &Qwen35Grads, f: impl Fn(usize, usize) -> f32) {
    for (p, (buf, off, n, dtype)) in bank_parts(g).into_iter().enumerate() {
        match dtype {
            DType::BF16 => {
                let mut m = buf.contents_u16();
                for (i, x) in m[off / 2..][..n].iter_mut().enumerate() {
                    let v = f(p, i);
                    *x = f32_to_bf16_bits(v);
                    assert_eq!(bf16_bits_to_f32(*x).to_bits(), v.to_bits(), "{v} is not a bf16");
                }
            }
            _ => {
                let mut m = buf.contents_f32();
                for (i, x) in m[off / 4..][..n].iter_mut().enumerate() {
                    *x = f(p, i);
                }
            }
        }
    }
}

/// The gradients' squares in f64, as `read_gradients` reads them.
fn host_squares(rt: &Arc<GpuRuntime>, model: &Qwen35Model, g: &Qwen35Grads) -> Vec<f64> {
    let table = model.parameter_table().unwrap();
    to_host_grads(rt, model, &table, g)
        .into_iter()
        .flatten()
        .map(|x| x * x)
        .collect()
}

fn to_host_grads(rt: &Arc<GpuRuntime>, model: &Qwen35Model, table: &[ParamInfo], g: &Qwen35Grads) -> Vec<Vec<f64>> {
    let ts = alloc_table(rt, table);
    model.read_gradients(g, &ts).unwrap();
    to_host(&ts)
}

/// `grad_sq_norm` is the exact sum of the squares rounded once to f64
/// wherever each piece of 4096 elements sums exactly in f32 in any order:
/// here every gradient is `m * 2^s` with `m` in `0..32` and a scale `s` per
/// parameter over `2^-14..2^6`, so a piece's squares are integers of one
/// unit below 2^24, while the whole sum spans ~2^62 and does not fit f32
/// (or f64: it rounds). The host forms it exactly in integers. On f32 and
/// bf16 banks of a model with a 64 000-element embedding (16 pieces, the
/// last short) beside the 64-wide rows of the matrices, both repeatable.
/// Then a real step's gradients: within 2^-18 relative of the f64 sum.
#[test]
fn grad_sq_norm_is_the_exact_sum_rounded_once() {
    let cfg = Qwen35Config {
        vocab: 1000,
        ..Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap()
    };
    for precision in [Precision::F32, Precision::Bf16] {
        let rt = GpuRuntime::new().unwrap();
        let model = Qwen35Model::random_tower(&rt, cfg.clone(), precision, 3).unwrap();
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        let value = |p: usize, i: usize| {
            let m = ((i as u64).wrapping_mul(0x9e37_79b9) >> 7) % 32;
            let s = -14 + ((p * 7) % 21) as i32;
            m as f32 * 2f32.powi(s - 10)
        };
        fill_bank(&bank, value);
        // Each square is an integer number of 2^-48 (the smallest scale,
        // squared, times 2^-20) below 2^53, so exact in f64 and in u128.
        let units: u128 = host_squares(&rt, &model, &bank)
            .iter()
            .map(|&q| {
                let u = q * 2f64.powi(48);
                assert_eq!(u.fract(), 0.0);
                u as u128
            })
            .sum();
        let want = units as f64 * 2f64.powi(-48);
        let got = model.grad_sq_norm(&bank).unwrap();
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{precision:?}: grad_sq_norm {got:e}, the exact sum rounded once {want:e}"
        );
        assert_eq!(
            model.grad_sq_norm(&bank).unwrap().to_bits(),
            got.to_bits(),
            "{precision:?}: rerun"
        );
        // A non-finite gradient is a non-finite norm.
        for (bad, check) in [
            (f32::INFINITY, f64::is_infinite as fn(f64) -> bool),
            (f32::NAN, f64::is_nan),
        ] {
            fill_bank(&bank, |p, i| if p == 3 && i == 5 { bad } else { value(p, i) });
            let got = model.grad_sq_norm(&bank).unwrap();
            assert!(check(got), "{precision:?}: a {bad} gradient gave {got}");
        }
    }

    let (rt, model) = load();
    let s = model.train_step(&ids(), GemmOperands::ExactF32).unwrap();
    let want: f64 = host_squares(&rt, &model, &s.grads).iter().sum();
    let got = model.grad_sq_norm(&s.grads).unwrap();
    assert!(
        (got - want).abs() <= 2f64.powi(-18) * want,
        "a step's grad_sq_norm {got:e} vs {want:e} in f64"
    );
}

/// Gradients whose squares leave f32's range are still summed exactly: every
/// gradient `m * 2^s` with `m` in `0..32` and one `s` for the whole bank, at
/// `s = 100` (each square near 2^210, past f32's 2^128) and `s = -100` (near
/// 2^-200, below f32's smallest subnormal), on f32 and bf16 banks. Each
/// piece's prescale is a power of two, so the result is the exact sum
/// `sum(m^2) * 2^(2s)` rounded once: finite and nonzero, where squaring
/// unscaled gave +inf and 0.
#[test]
fn grad_sq_norm_of_huge_and_tiny_gradients_is_exact() {
    let cfg = Qwen35Config {
        vocab: 1000,
        ..Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap()
    };
    for precision in [Precision::F32, Precision::Bf16] {
        let rt = GpuRuntime::new().unwrap();
        let model = Qwen35Model::random_tower(&rt, cfg.clone(), precision, 3).unwrap();
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        let m = |i: usize| ((i as u64).wrapping_mul(0x9e37_79b9) >> 7) % 32;
        for s in [100i32, -100] {
            fill_bank(&bank, |_, i| m(i) as f32 * 2f32.powi(s));
            // m^2 < 2^10, so the integer sum is exact in u128 and its f64
            // rounding times 2^(2s) is the exact sum rounded once.
            let units: u128 = host_squares(&rt, &model, &bank)
                .iter()
                .map(|&q| {
                    let u = q * 2f64.powi(-2 * s);
                    assert_eq!(u.fract(), 0.0);
                    u as u128
                })
                .sum();
            let want = units as f64 * 2f64.powi(2 * s);
            assert!(want.is_finite() && want > 0.0);
            let got = model.grad_sq_norm(&bank).unwrap();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{precision:?} at 2^{s}: grad_sq_norm {got:e}, the exact sum rounded once {want:e}"
            );
        }
    }
}

/// Per-entry moment copies are the whole-table copies' bits for each moment
/// storage, a write of a read is the same state (8-bit included), a write
/// lands in its entry alone, and a bad index or length or a busy runtime is
/// refused with nothing written.
#[test]
fn per_entry_moment_host_copies_match_the_whole_table() {
    for moments in [MomentStorage::F32, MomentStorage::Bf16, MomentStorage::Block8] {
        let (rt, model) = load_bf16();
        let table = model.parameter_table().unwrap();
        let wd = model.default_weight_decay(0.1).unwrap();
        let mut state = AdamW::with_config(
            &model,
            AdamWConfig {
                update: UpdateRule::F32Master,
                moments,
            },
        )
        .unwrap();
        for _ in 0..2 {
            let s = model.train_step(&ids(), GemmOperands::Bf16).unwrap();
            model.adamw_step(&s.grads, &mut state, &hyper(), &wd).unwrap();
        }
        let whole = |state: &AdamW, which: Moment| -> Vec<Vec<u32>> {
            let ts = alloc_table(&rt, &table);
            model.read_adamw_moment(state, which, &ts).unwrap();
            ts.iter()
                .map(|t| t.read_f32().unwrap().iter().map(|x| x.to_bits()).collect())
                .collect()
        };
        for which in [Moment::First, Moment::Second] {
            let want = whole(&state, which);
            assert!(want.iter().any(|v| v.iter().any(|&b| f32::from_bits(b) != 0.0)));
            for (i, p) in table.iter().enumerate() {
                let n: usize = p.shape.iter().product();
                let mut got = vec![0.0f32; n];
                model.read_adamw_moment_host(&state, which, i, &mut got).unwrap();
                let got: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
                assert_eq!(got, want[i], "{moments:?} {which:?} {}", p.name);
                // Write of a read: the same state.
                let vals: Vec<f32> = got.iter().map(|&b| f32::from_bits(b)).collect();
                model.write_adamw_moment_host(&mut state, which, i, &vals).unwrap();
                assert_eq!(
                    whole(&state, which),
                    want,
                    "{moments:?} {which:?} {}: rewrite moved state",
                    p.name
                );
            }
            // New values land in one entry alone (read back through the same codec).
            let i = table.len() / 2;
            let n: usize = table[i].shape.iter().product();
            let new: Vec<f32> = (0..n).map(|k| 1e-3 * (1 + k % 17) as f32).collect();
            model.write_adamw_moment_host(&mut state, which, i, &new).unwrap();
            let after = whole(&state, which);
            for j in 0..table.len() {
                assert_eq!(after[j] == want[j], j != i, "{moments:?} {which:?}: entry {j}");
            }
            let mut back = vec![0.0f32; n];
            model.read_adamw_moment_host(&state, which, i, &mut back).unwrap();
            assert_eq!(back.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), after[i]);
            // f32 is exact; bf16 is within 2^-8 relative and an 8-bit code
            // within 0.0081 of its block's largest magnitude (here 0.017).
            let tol = if moments == MomentStorage::F32 {
                0.0
            } else {
                0.0081 * 0.017
            };
            for (b, w) in back.iter().zip(&new) {
                assert!((b - w).abs() <= tol, "{moments:?} {which:?}: {b} vs {w}");
            }
        }
        let n: usize = table[0].shape.iter().product();
        let e = |r: Result<(), String>, needle: &str| {
            let m = r.expect_err(needle);
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        };
        let before = whole(&state, Moment::First);
        let len = table.len();
        e(
            model.read_adamw_moment_host(&state, Moment::First, len, &mut vec![0.0; n]),
            "out of range",
        );
        e(
            model.write_adamw_moment_host(&mut state, Moment::First, 0, &vec![1.0; n - 1]),
            "takes",
        );
        e(
            model.write_adamw_moment_host(&mut state, Moment::First, 0, &[]),
            "takes",
        );
        {
            let other = rt.alloc_tensor_f32(&[1]).unwrap();
            let _held = other.buffer.try_contents_f32().unwrap();
            e(
                model.read_adamw_moment_host(&state, Moment::First, 0, &mut vec![0.0; n]),
                "runtime busy",
            );
            e(
                model.write_adamw_moment_host(&mut state, Moment::First, 0, &vec![1.0; n]),
                "runtime busy",
            );
        }
        let (_, other_model) = load_bf16();
        let other_state = AdamW::with_config(
            &other_model,
            AdamWConfig {
                update: UpdateRule::F32Master,
                moments,
            },
        )
        .unwrap();
        e(
            model.read_adamw_moment_host(&other_state, Moment::First, 0, &mut vec![0.0; n]),
            "",
        );
        assert_eq!(whole(&state, Moment::First), before, "{moments:?}: a refusal wrote");
    }
}

// ---- adversarial: per-entry moment host copies ----

fn awkward_bits(n: usize, seed: u32) -> Vec<f32> {
    const FIXED: [u32; 20] = [
        0x3f80_8000,
        0x3f81_8000,
        0x3f80_8001,
        0xbf80_8000,
        0x7f7f_8000,
        0x7f7f_ffff,
        0xff7f_8000,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fc0_0001,
        0xffff_ffff,
        0x0000_0001,
        0x0000_8000,
        0x0001_8000,
        0x8000_0001,
        0x0000_0000,
        0x8000_0000,
        0x007f_ffff,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            f32::from_bits(if i < FIXED.len() {
                FIXED[i]
            } else {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                s
            })
        })
        .collect()
}

/// A bf16 moment written on the host stores the bits the GPU
/// `write_adamw_moment` stores for the same f32 input (ties, overflow, NaN
/// payloads, subnormals).
#[test]
fn bf16_moment_host_write_rounds_like_the_gpu_write_bit_for_bit() {
    let (rt, model) = load_bf16();
    let table = model.parameter_table().unwrap();
    let cfg = AdamWConfig {
        update: UpdateRule::F32Master,
        moments: MomentStorage::Bf16,
    };
    let mut gpu = AdamW::with_config(&model, cfg).unwrap();
    let mut host = AdamW::with_config(&model, cfg).unwrap();
    for which in [Moment::First, Moment::Second] {
        let vals: Vec<Vec<f32>> = table
            .iter()
            .enumerate()
            .map(|(i, p)| awkward_bits(p.shape.iter().product(), 0x85eb_ca6b ^ i as u32))
            .collect();
        let ts = alloc_table(&rt, &table);
        for (t, v) in ts.iter().zip(&vals) {
            t.write_f32(v).unwrap();
        }
        model.write_adamw_moment(&mut gpu, which, &ts).unwrap();
        for (i, v) in vals.iter().enumerate() {
            model.write_adamw_moment_host(&mut host, which, i, v).unwrap();
            let n = v.len();
            let (mut a, mut b) = (vec![0.0f32; n], vec![0.0f32; n]);
            model.read_adamw_moment_host(&gpu, which, i, &mut a).unwrap();
            model.read_adamw_moment_host(&host, which, i, &mut b).unwrap();
            let diff: Vec<(usize, u32, u32, u32)> = (0..n)
                .filter(|&k| a[k].to_bits() != b[k].to_bits())
                .map(|k| (k, v[k].to_bits(), a[k].to_bits(), b[k].to_bits()))
                .collect();
            assert!(
                diff.is_empty(),
                "{which:?} {}: {} of {n} differ; (index, input, gpu, host) {diff:x?}",
                table[i].name,
                diff.len()
            );
        }
    }
}

/// An unwaited step is waited for by the per-entry reads (parameters and each
/// moment storage), so they see the stepped values, not stale ones.
#[test]
fn per_entry_host_reads_wait_for_an_unwaited_step() {
    for moments in [MomentStorage::F32, MomentStorage::Bf16, MomentStorage::Block8] {
        let (rt, model) = load_bf16();
        let table = model.parameter_table().unwrap();
        let wd = model.default_weight_decay(0.1).unwrap();
        let cfg = AdamWConfig {
            update: UpdateRule::F32Master,
            moments,
        };
        let mut state = AdamW::with_config(&model, cfg).unwrap();
        let s = model.train_step(&ids(), GemmOperands::Bf16).unwrap();
        let i = table.len() / 2;
        let n: usize = table[i].shape.iter().product();
        let mut before = vec![0.0f32; n];
        model.read_parameter_host(i, &mut before).unwrap();
        model.adamw_step_unwaited(&s.grads, &mut state, &hyper(), &wd).unwrap();
        let mut got = vec![0.0f32; n];
        let mut m = vec![0.0f32; n];
        // The very first host read after the unwaited encode.
        model.read_adamw_moment_host(&state, Moment::First, i, &mut m).unwrap();
        model.read_parameter_host(i, &mut got).unwrap();
        let ts = alloc_table(&rt, &table);
        model.read_parameters(&ts).unwrap();
        let want = ts[i].read_f32().unwrap();
        assert_ne!(got, before, "{moments:?}: the step did not move the entry");
        assert_eq!(got, want, "{moments:?}: a host read ran before the step finished");
        let ms = alloc_table(&rt, &table);
        model.read_adamw_moment(&state, Moment::First, &ms).unwrap();
        assert_eq!(m, ms[i].read_f32().unwrap(), "{moments:?}: moment read raced the step");
    }
}

/// On a poisoned runtime every per-entry moment call (the 8-bit one
/// dispatches a kernel) is an `Err`, never a panic, and the state's step
/// count does not move.
#[test]
fn per_entry_moment_host_calls_refuse_on_a_poisoned_runtime() {
    for moments in [MomentStorage::F32, MomentStorage::Bf16, MomentStorage::Block8] {
        let (rt, model) = load_bf16();
        let table = model.parameter_table().unwrap();
        let cfg = AdamWConfig {
            update: UpdateRule::F32Master,
            moments,
        };
        let mut state = AdamW::with_config(&model, cfg).unwrap();
        let n: usize = table[1].shape.iter().product();
        rt.poison_as_shared_event_timeout_for_test();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut buf = vec![0.0f32; n];
            let a = model.read_adamw_moment_host(&state, Moment::First, 1, &mut buf);
            let b = model.write_adamw_moment_host(&mut state, Moment::Second, 1, &vec![1.0; n]);
            (a, b)
        }));
        let (a, b) = r.unwrap_or_else(|_| panic!("{moments:?}: a poisoned runtime panicked"));
        assert!(a.is_err(), "{moments:?}: read succeeded on a poisoned runtime");
        assert!(b.is_err(), "{moments:?}: write succeeded on a poisoned runtime");
    }
}

// ---- the fused step: AdamW applied as the backward runs ---------------------

/// Every parameter, both moments and any auxiliary state as raw bits (the
/// f32 widening of a bf16 value is exact, so equal bits are equal storage).
fn state_bits(rt: &Arc<GpuRuntime>, model: &Qwen35Model, table: &[ParamInfo], s: &AdamW) -> Vec<Vec<u32>> {
    let (p, m, v, x) = (
        alloc_table(rt, table),
        alloc_table(rt, table),
        alloc_table(rt, table),
        alloc_table(rt, table),
    );
    model.read_parameters(&p).unwrap();
    model.read_adamw_moment(s, Moment::First, &m).unwrap();
    model.read_adamw_moment(s, Moment::Second, &v).unwrap();
    let mut all = vec![p, m, v];
    if model.read_adamw_aux(s, &x).is_ok() {
        all.push(x);
    }
    all.iter()
        .flatten()
        .map(|t| t.read_f32().unwrap().iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// `STEPS` steps of the fused step against the bank's `train_step_into` then
/// `adamw_step`, on identical inputs from identical weights: the loss, every
/// parameter, every moment and any auxiliary state are the same bits after
/// each step. The fused arm runs with unzeroed allocations poisoned to NaN,
/// so a scratch read before it is written shows as NaN in what follows.
fn fused_equals_bank(precision: Precision, config: AdamWConfig, mm: GemmOperands) {
    const STEPS: usize = 3;
    let load_as = |p: Precision| match p {
        Precision::F32 => load(),
        Precision::Bf16 => load_bf16(),
    };
    let (rt_a, a) = load_as(precision);
    let (rt_b, b) = load_as(precision);
    rt_b.set_poison_unzeroed(true);
    let table = a.parameter_table().unwrap();
    let wd = a.default_weight_decay(0.1).unwrap();
    let (mut sa, mut sb) = (
        AdamW::with_config(&a, config).unwrap(),
        AdamW::with_config(&b, config).unwrap(),
    );
    let bank = Qwen35Grads::zeros_like(&a).unwrap();
    let mut ids = ids();
    for step in 1..=STEPS {
        // A different sequence each step.
        ids.rotate_left(1);
        let la = a.train_step_into(&ids, mm, Supervise::Causal, &bank, false).unwrap();
        a.adamw_step(&bank, &mut sa, &hyper(), &wd).unwrap();
        let lb = b
            .train_step_fused_adamw(&ids, mm, Supervise::Causal, &mut sb, &hyper(), &wd)
            .unwrap();
        assert_eq!(la.to_bits(), lb.to_bits(), "{config} step {step}: loss {la} vs {lb}");
        assert_eq!(sb.step_count(), step as u64);
        let (want, got) = (state_bits(&rt_a, &a, &table, &sa), state_bits(&rt_b, &b, &table, &sb));
        assert_eq!(want.len(), got.len(), "{config} step {step}: state kinds differ");
        for (k, (w, g)) in want.iter().zip(&got).enumerate() {
            let (i, which) = (
                k % table.len(),
                ["parameter", "first moment", "second moment", "aux"][k / table.len()],
            );
            assert!(
                w == g,
                "{config} step {step}: {} {which} differs from the bank's step",
                table[i].name
            );
        }
    }
}

#[test]
fn a_fused_f32_step_is_the_banks_step_bit_for_bit() {
    fused_equals_bank(Precision::F32, AdamWConfig::F32, GemmOperands::ExactF32);
}

#[test]
fn a_fused_bf16_step_is_the_banks_step_bit_for_bit() {
    // The documented default: stochastic rounding, f32 moments. Then the
    // other state shapes (an f32 master, a Kahan compensation, 8-bit moments).
    for config in [
        AdamWConfig {
            update: UpdateRule::bf16_default(11),
            moments: MomentStorage::F32,
        },
        AdamWConfig {
            update: UpdateRule::F32Master,
            moments: MomentStorage::Bf16,
        },
        AdamWConfig {
            update: UpdateRule::Bf16Kahan,
            moments: MomentStorage::Block8,
        },
    ] {
        fused_equals_bank(Precision::Bf16, config, GemmOperands::Bf16);
    }
}

/// A forward taken before a fused step is of weights the step changed: the
/// generation guard refuses its backward, as it does after `adamw_step`.
#[test]
fn a_forward_pending_before_a_fused_step_is_refused_after_it() {
    let (_rt, model) = load();
    let wd = model.default_weight_decay(0.1).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let ids = ids();
    let pending = model
        .train_forward(&ids, GemmOperands::ExactF32, Supervise::Causal)
        .unwrap();
    model
        .train_step_fused_adamw(
            &ids,
            GemmOperands::ExactF32,
            Supervise::Causal,
            &mut state,
            &hyper(),
            &wd,
        )
        .unwrap();
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let err = model.train_backward_into(pending, None, &bank, false).unwrap_err();
    assert!(
        err.contains("parameters were written after the step's forward"),
        "{err}"
    );
}

/// What a fused step cannot do is refused before anything is written: a clip
/// coefficient (the global norm does not exist mid-backward), a decay table
/// of the wrong size, a state made for another model.
#[test]
fn fused_step_refusals_move_nothing() {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let ids = ids();
    let before = state_bits(&rt, &model, &table, &state);
    let e = |r: Result<f64, String>, needle: &str| {
        let m = r.err().unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    let step = |state: &mut AdamW, hyper: AdamWHyper, wd: &[f32]| {
        model.train_step_fused_adamw(&ids, GemmOperands::ExactF32, Supervise::Causal, state, &hyper, wd)
    };
    e(
        step(
            &mut state,
            AdamWHyper {
                grad_scale: 0.5,
                ..hyper()
            },
            &wd,
        ),
        "clip coefficient",
    );
    e(step(&mut state, hyper(), &wd[1..]), "weight decays for");
    let mut bad = wd.clone();
    bad[2] = f32::NAN;
    e(step(&mut state, hyper(), &bad), "weight decay NaN");
    e(
        step(
            &mut state,
            AdamWHyper {
                lr: f64::NAN,
                ..hyper()
            },
            &wd,
        ),
        "lr NaN",
    );
    // A one-token sequence has no prediction: refused by the forward, before
    // the first update.
    e(
        model.train_step_fused_adamw(
            &ids[..1],
            GemmOperands::ExactF32,
            Supervise::Causal,
            &mut state,
            &hyper(),
            &wd,
        ),
        "at least two tokens",
    );
    assert_eq!(state.step_count(), 0);
    assert!(
        state_bits(&rt, &model, &table, &state) == before,
        "a refused fused step moved the model or the state"
    );
}

// ---- adversarial lane for the fused step (commit 8bf6214) --------------------
//
// Written against the tiny fixtures, built but not run by the lane that wrote
// them (a locked job runs device tests). The ones that assert a behavior the
// fused step does not have yet are `#[ignore]`d with the break they expose.

fn dir_untied() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train_untied")
}

fn dir_grouped() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train_grouped")
}

fn ids_in(dir: &Path) -> Vec<u32> {
    read_npy(&dir.join("ids.npy"))
        .unwrap()
        .i64_slice()
        .unwrap()
        .iter()
        .map(|&x| x as u32)
        .collect()
}

/// A fixture as the trainer loads it: the f32 model whole, the bf16 one as a
/// tower (no packed head copy).
fn load_dir(dir: &Path, precision: Precision) -> (Arc<GpuRuntime>, Qwen35Model) {
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let model = match precision {
        Precision::F32 => Qwen35Model::load(&rt, &st, "model.", cfg, precision),
        Precision::Bf16 => Qwen35Model::load_tower(&rt, &st, "model.", cfg, precision),
    }
    .unwrap();
    (rt, model)
}

/// One step's supervision, owned so a test can list several.
#[derive(Clone)]
enum Sup {
    Causal,
    Rows {
        positions: Vec<u32>,
        targets: Vec<u32>,
        scale: f32,
    },
}

impl Sup {
    fn rows(positions: &[u32], targets: &[u32], scale: f32) -> Self {
        Sup::Rows {
            positions: positions.to_vec(),
            targets: targets.to_vec(),
            scale,
        }
    }

    fn get(&self) -> Supervise<'_> {
        match self {
            Sup::Causal => Supervise::Causal,
            Sup::Rows {
                positions,
                targets,
                scale,
            } => Supervise::Rows {
                positions,
                targets,
                scale: *scale,
            },
        }
    }
}

/// Every supervision shape a fused step must take: the causal loss, scored
/// rows (a subset, in a non-monotone order), one row, and no row at all (the
/// head adds nothing, the update still runs on zero gradients).
fn sups_for(ids: &[u32]) -> Vec<Sup> {
    let n = ids.len() as u32;
    assert!(n >= 5, "the fixture's sequence is too short for the positions below");
    vec![
        Sup::Causal,
        Sup::rows(&[n - 1, 1, 3], &[ids[0], ids[2], ids[4]], 0.5),
        Sup::rows(&[2], &[ids[1]], 1.0),
        Sup::rows(&[], &[], 1.0),
    ]
}

/// Fill the pool's freelist with NaN-filled buffers of every parameter's
/// size (f32 and bf16), so a `GpuRuntime::alloc_buffer` that is not zeroed
/// (the fused step's raw-vector scratch, which `set_poison_unzeroed` does not
/// reach) hands the step garbage it must overwrite before anything reads it.
fn dirty_the_pool(rt: &Arc<GpuRuntime>, table: &[ParamInfo]) {
    let junk: Vec<GpuBuffer> = table
        .iter()
        .flat_map(|p| {
            let n: usize = p.shape.iter().product();
            [n * 4, n * 2]
        })
        .map(|bytes| {
            let b = rt.alloc_buffer(bytes.div_ceil(4).max(1) * 4).unwrap();
            b.try_write_u32(&vec![0x7fc0_0000; b.nbytes() / 4]).unwrap();
            b
        })
        .collect();
    drop(junk);
    rt.synchronize().unwrap();
}

/// A different decay for every table entry, so a layer's update that read the
/// decay table at the wrong offset shows (the default table has two values).
fn distinct_decay(table: &[ParamInfo]) -> Vec<f32> {
    (0..table.len()).map(|i| 0.002 * (i as f32 + 1.0)).collect()
}

/// [`fused_equals_bank`] over any fixture, every supervision of
/// [`sups_for`] (twice round), a distinct decay per entry and, when `dirty`,
/// a pool full of NaN buffers for the fused arm.
fn fused_equals_bank_on(dir: &Path, precision: Precision, config: AdamWConfig, mm: GemmOperands, dirty: bool) {
    let (rt_a, a) = load_dir(dir, precision);
    let (rt_b, b) = load_dir(dir, precision);
    rt_b.set_poison_unzeroed(true);
    let table = a.parameter_table().unwrap();
    if dirty {
        dirty_the_pool(&rt_b, &table);
    }
    let wd = distinct_decay(&table);
    let (mut sa, mut sb) = (
        AdamW::with_config(&a, config).unwrap(),
        AdamW::with_config(&b, config).unwrap(),
    );
    let bank = Qwen35Grads::zeros_like(&a).unwrap();
    let mut ids = ids_in(dir);
    let sups = sups_for(&ids);
    for (step, sup) in sups.iter().chain(sups.iter()).enumerate() {
        ids.rotate_left(1);
        let la = a.train_step_into(&ids, mm, sup.get(), &bank, false).unwrap();
        a.adamw_step(&bank, &mut sa, &hyper(), &wd).unwrap();
        let lb = b
            .train_step_fused_adamw(&ids, mm, sup.get(), &mut sb, &hyper(), &wd)
            .unwrap();
        assert_eq!(
            la.to_bits(),
            lb.to_bits(),
            "{} {config} step {step}: loss {la} vs {lb}",
            dir.display()
        );
        assert!(lb.is_finite(), "step {step}: loss {lb}");
        assert_eq!(sb.step_count(), step as u64 + 1, "once per step, not per layer");
        let (want, got) = (state_bits(&rt_a, &a, &table, &sa), state_bits(&rt_b, &b, &table, &sb));
        assert_eq!(want.len(), got.len());
        for (k, (w, g)) in want.iter().zip(&got).enumerate() {
            let (i, which) = (
                k % table.len(),
                ["parameter", "first moment", "second moment", "aux"][k / table.len()],
            );
            assert!(
                w == g,
                "{} {config} step {step}: {} {which} differs from the bank's step",
                dir.display(),
                table[i].name
            );
        }
    }
}

/// Attack 2, 3: the untied head (its own update after the gather) and the
/// grouped GDN heads, on f32 and on two bf16 state shapes, with every
/// supervision shape, a distinct decay per entry and a dirty pool.
#[test]
fn fused_equals_bank_on_the_untied_and_grouped_fixtures() {
    let bf16_configs = [
        AdamWConfig {
            update: UpdateRule::bf16_default(11),
            moments: MomentStorage::F32,
        },
        AdamWConfig {
            update: UpdateRule::Bf16Kahan,
            moments: MomentStorage::Block8,
        },
    ];
    for dir in [dir_untied(), dir_grouped()] {
        fused_equals_bank_on(&dir, Precision::F32, AdamWConfig::F32, GemmOperands::ExactF32, true);
        for config in bf16_configs {
            fused_equals_bank_on(&dir, Precision::Bf16, config, GemmOperands::Bf16, true);
        }
    }
}

/// Attack 2: the tied fixture under the supervision shapes, a distinct decay
/// per entry and a dirty pool (the existing parity tests run `Causal` and a
/// two-valued decay).
#[test]
fn fused_equals_bank_on_the_tied_fixture_for_every_supervision() {
    let dir = fixture();
    fused_equals_bank_on(&dir, Precision::F32, AdamWConfig::F32, GemmOperands::ExactF32, true);
    fused_equals_bank_on(
        &dir,
        Precision::Bf16,
        AdamWConfig {
            update: UpdateRule::bf16_default(3),
            moments: MomentStorage::F32,
        },
        GemmOperands::Bf16,
        true,
    );
}

/// Attack 3: `parameter_table()` (what `slots` builds, and so what
/// `adamw_step` and the fused step index by) is transformers' order and names
/// on the tied, untied and grouped fixtures, f32 and bf16. The oracle is
/// written from the config, not from `slots`.
#[test]
fn the_parameter_table_keeps_its_order_and_names() {
    use tessl::qwen35_model::LayerKind;
    for dir in [fixture(), dir_untied(), dir_grouped()] {
        for precision in [Precision::F32, Precision::Bf16] {
            let (_rt, model) = load_dir(&dir, precision);
            let cfg = model.config();
            let mut want: Vec<String> = vec!["embed_tokens.weight".into()];
            for (l, kind) in cfg.layers.iter().enumerate() {
                let mixer: &[&str] = match kind {
                    LayerKind::LinearAttention => &[
                        "linear_attn.in_proj_qkv.weight",
                        "linear_attn.in_proj_z.weight",
                        "linear_attn.in_proj_b.weight",
                        "linear_attn.in_proj_a.weight",
                        "linear_attn.out_proj.weight",
                        "linear_attn.conv1d.weight",
                        "linear_attn.A_log",
                        "linear_attn.dt_bias",
                        "linear_attn.norm.weight",
                    ],
                    LayerKind::FullAttention => &[
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight",
                        "self_attn.q_norm.weight",
                        "self_attn.k_norm.weight",
                    ],
                };
                want.extend(mixer.iter().map(|s| format!("layers.{l}.{s}")));
                want.extend(
                    [
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                        "mlp.down_proj.weight",
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                    ]
                    .iter()
                    .map(|s| format!("layers.{l}.{s}")),
                );
            }
            want.push("norm.weight".into());
            if !cfg.tie_word_embeddings {
                want.push("lm_head.weight".into());
            }
            let got: Vec<String> = model.parameter_table().unwrap().into_iter().map(|p| p.name).collect();
            assert_eq!(got, want, "{} {precision:?}", dir.display());
            // The default decay table follows the table by position.
            let wd = model.default_weight_decay(0.1).unwrap();
            assert_eq!(wd.len(), got.len());
            for (name, d) in got.iter().zip(&wd) {
                assert_eq!(*d == 0.0, excluded_from_weight_decay(name), "{name}");
            }
        }
    }
}

/// Attack 4: a fused step that fails in a layer's rebuild after the layers
/// above it were updated. Pins what holds: the error comes back, the
/// generation moved (a forward taken before is refused), layers above the
/// failure are updated and layers below, the embedding and the final norm are
/// not. `fixture()` has two layers; the hook counts a forward and a rebuild
/// per layer, so 3 lets the forward and the top layer's rebuild pass and
/// fails layer 0's rebuild, after layer 1's update was encoded.
#[test]
fn a_fused_step_failing_in_a_rebuild_leaves_the_top_updated_and_the_generation_moved() {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let ids = ids();
    let pending = model
        .train_forward(&ids, GemmOperands::ExactF32, Supervise::Causal)
        .unwrap();
    let before = state_bits(&rt, &model, &table, &state);
    model.fail_layer_after_for_test(3);
    let err = model
        .train_step_fused_adamw(
            &ids,
            GemmOperands::ExactF32,
            Supervise::Causal,
            &mut state,
            &hyper(),
            &wd,
        )
        .unwrap_err();
    assert!(err.contains("injected fault"), "{err}");
    rt.synchronize().unwrap();
    let after = state_bits(&rt, &model, &table, &state);
    // `state_bits` lists every table entry's parameter first.
    for (i, p) in table.iter().enumerate() {
        let moved = before[i] != after[i];
        let above = p.name.starts_with("layers.1.");
        assert_eq!(moved, above, "{}: moved {moved}, in an updated layer {above}", p.name);
    }
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let refused = model.train_backward_into(pending, None, &bank, false).unwrap_err();
    assert!(refused.contains("parameters were written"), "{refused}");
}

/// Attack 4, the break: after that failure the model is half updated and
/// nothing says so. `state.step_count()` reports a whole step (1) although
/// layer 0, the embedding and the final norm have no moment history, the
/// runtime is not poisoned, and the next fused step runs on at count 2, so
/// the half that was skipped is updated at a bias correction for step 2 with
/// moments that have seen no step. Wanted: the failed step leaves a marker
/// (poison the runtime or the state, or refuse the next step), or the count
/// does not claim the step. Fixed: the model and the state latch as torn.
#[test]
fn a_fused_step_failing_midway_is_not_continued_silently() {
    let (rt, model) = load();
    let wd = model.default_weight_decay(0.1).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let ids = ids();
    model.fail_layer_after_for_test(3);
    let first = model.train_step_fused_adamw(
        &ids,
        GemmOperands::ExactF32,
        Supervise::Causal,
        &mut state,
        &hyper(),
        &wd,
    );
    assert!(first.is_err());
    let again = model.train_step_fused_adamw(
        &ids,
        GemmOperands::ExactF32,
        Supervise::Causal,
        &mut state,
        &hyper(),
        &wd,
    );
    assert!(
        rt.is_poisoned() || state.step_count() == 0 || again.is_err(),
        "the model is half updated (step count {}), the runtime is not poisoned, and the next step ran",
        state.step_count()
    );
    // The refusal names the torn step, and covers every way to use the
    // half-updated weights: a fused step, a bank step, an optimizer step, a
    // forward, a train forward; and the state refuses a reloaded model.
    let torn = again.unwrap_err();
    assert!(torn.contains("failed part way") && torn.contains("reload"), "{torn}");
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    let refusals = [
        model
            .train_step_into(&ids, GemmOperands::ExactF32, Supervise::Causal, &bank, false)
            .map(|_| ()),
        model.adamw_step(&bank, &mut state, &hyper(), &wd),
        model.forward(&ids, false).map(|_| ()),
        model
            .train_forward(&ids, GemmOperands::ExactF32, Supervise::Causal)
            .map(|_| ()),
    ];
    for (i, r) in refusals.into_iter().enumerate() {
        let e = r.err().unwrap_or_else(|| panic!("entry {i} ran on a torn model"));
        assert!(e.contains("failed part way"), "entry {i}: {e}");
    }
    let (_rt2, fresh) = load();
    let e = fresh
        .train_step_fused_adamw(
            &ids,
            GemmOperands::ExactF32,
            Supervise::Causal,
            &mut state,
            &hyper(),
            &wd,
        )
        .unwrap_err();
    assert!(
        e.contains("failed part way"),
        "a torn state ran on a reloaded model: {e}"
    );
}

/// Attack 6, the break: a fused step has no moment between the loss and the
/// first update where a caller can look at the gradient (a bank step lets
/// the caller test `grad_sq_norm` and skip `adamw_step`), so a non-finite
/// loss, known after the forward, must be refused before the backward, or the
/// update writes NaN into every parameter and moment. Wanted: an error and
/// nothing moved. Fixed: refused after the forward, before any update.
#[test]
fn a_fused_step_on_a_non_finite_loss_moves_nothing() {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let at = table.iter().position(|p| p.name == "norm.weight").unwrap();
    let n: usize = table[at].shape.iter().product();
    model.write_parameter_host(at, &vec![f32::INFINITY; n]).unwrap();
    let before = state_bits(&rt, &model, &table, &state);
    let out = model.train_step_fused_adamw(
        &ids(),
        GemmOperands::ExactF32,
        Supervise::Causal,
        &mut state,
        &hyper(),
        &wd,
    );
    rt.synchronize().unwrap();
    assert!(out.is_err(), "a non-finite loss was accepted: {out:?}");
    assert!(state_bits(&rt, &model, &table, &state) == before, "it moved the model");
    assert_eq!(state.step_count(), 0);
}

/// A hostile call: what it is, its tokens, supervision, hyperparameters and
/// decay table.
type Hostile<'a> = (&'a str, &'a [u32], Sup, AdamWHyper, &'a [f32]);

/// Attack 6: every hostile input is refused with nothing moved: empty and
/// out-of-range tokens, a position twice or past the tokens, a target past
/// the vocabulary, a non-finite row scale, a NaN clip scale, an infinite or
/// negative decay, a decay table one too long.
#[test]
fn fused_step_refuses_hostile_inputs_and_moves_nothing() {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let ids = ids();
    let vocab = model.config().vocab;
    let n = ids.len() as u32;
    let before = state_bits(&rt, &model, &table, &state);
    let mut bad_token = ids.clone();
    bad_token[2] = vocab;
    let mut inf_decay = wd.clone();
    *inf_decay.last_mut().unwrap() = f32::INFINITY;
    let mut negative_decay = wd.clone();
    negative_decay[0] = -0.5;
    let mut long_decay = wd.clone();
    long_decay.push(0.0);
    let nan_clip = AdamWHyper {
        grad_scale: f64::NAN,
        ..hyper()
    };
    let cases: Vec<Hostile<'_>> = vec![
        ("empty ids, causal", &[], Sup::Causal, hyper(), &wd),
        ("empty ids, rows", &[], Sup::rows(&[], &[], 1.0), hyper(), &wd),
        (
            "an id past the vocabulary, causal",
            &bad_token,
            Sup::Causal,
            hyper(),
            &wd,
        ),
        (
            "a target past the vocabulary",
            &ids,
            Sup::rows(&[1], &[vocab], 1.0),
            hyper(),
            &wd,
        ),
        ("a position twice", &ids, Sup::rows(&[1, 1], &[0, 0], 1.0), hyper(), &wd),
        (
            "a position past the tokens",
            &ids,
            Sup::rows(&[n], &[0], 1.0),
            hyper(),
            &wd,
        ),
        ("a NaN row scale", &ids, Sup::rows(&[1], &[0], f32::NAN), hyper(), &wd),
        (
            "an infinite row scale",
            &ids,
            Sup::rows(&[1], &[0], f32::INFINITY),
            hyper(),
            &wd,
        ),
        ("a NaN clip scale", &ids, Sup::Causal, nan_clip, &wd),
        ("an infinite decay", &ids, Sup::Causal, hyper(), &inf_decay),
        ("a negative decay", &ids, Sup::Causal, hyper(), &negative_decay),
        ("a decay table one long", &ids, Sup::Causal, hyper(), &long_decay),
        ("a decay table empty", &ids, Sup::Causal, hyper(), &[]),
    ];
    for (what, ids, sup, h, w) in cases {
        let out = model.train_step_fused_adamw(ids, GemmOperands::ExactF32, sup.get(), &mut state, &h, w);
        assert!(out.is_err(), "{what}: accepted");
        rt.synchronize().unwrap();
        assert_eq!(state.step_count(), 0, "{what}: the count moved");
        assert!(
            state_bits(&rt, &model, &table, &state) == before,
            "{what}: the model or the state moved"
        );
    }
}

/// Attack 6: a state made for another model (other slots: the untied one) or
/// on another runtime (the same shapes) is refused with nothing moved, and a
/// bf16 model refuses an f32-config state.
#[test]
fn fused_step_refuses_a_state_that_is_not_its_models() {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(0.1).unwrap();
    let ids = ids();
    let before = {
        let state = AdamW::new(&model).unwrap();
        state_bits(&rt, &model, &table, &state)
    };
    let (_rt_u, untied) = load_dir(&dir_untied(), Precision::F32);
    let (_rt_o, other_runtime) = load();
    let mut foreign = [AdamW::new(&untied).unwrap(), AdamW::new(&other_runtime).unwrap()];
    for (k, state) in foreign.iter_mut().enumerate() {
        let out = model.train_step_fused_adamw(&ids, GemmOperands::ExactF32, Supervise::Causal, state, &hyper(), &wd);
        assert!(out.is_err(), "foreign state {k} accepted");
        assert_eq!(state.step_count(), 0);
    }
    rt.synchronize().unwrap();
    let again = AdamW::new(&model).unwrap();
    assert!(
        state_bits(&rt, &model, &table, &again) == before,
        "a refusal moved the model"
    );

    // A bf16 model with a state of the f32 config: refused at creation, or
    // by the step before it writes.
    let (rt16, bf16) = load_dir(&fixture(), Precision::Bf16);
    let t16 = bf16.parameter_table().unwrap();
    if let Ok(mut f32_state) = AdamW::with_config(&bf16, AdamWConfig::F32) {
        let b0 = state_bits(&rt16, &bf16, &t16, &f32_state);
        let out = bf16.train_step_fused_adamw(
            &ids,
            GemmOperands::Bf16,
            Supervise::Causal,
            &mut f32_state,
            &hyper(),
            &bf16.default_weight_decay(0.1).unwrap(),
        );
        assert!(out.is_err(), "an f32-config state on a bf16 model was accepted");
        assert_eq!(f32_state.step_count(), 0);
        assert!(state_bits(&rt16, &bf16, &t16, &f32_state) == b0);
    }
}

/// Attack 1, async encode and dispatch batching: the update of layer L is
/// encoded behind a barrier on the queue, so a step encoded in one async
/// scope (every dispatch queued, one wait at the end) is the serialized
/// step's bits (every dispatch committed and waited: nothing can overlap).
#[test]
fn a_fused_step_in_an_async_scope_is_the_serialized_steps_bits() {
    for (dir, precision, config, mm) in [
        (fixture(), Precision::F32, AdamWConfig::F32, GemmOperands::ExactF32),
        (dir_untied(), Precision::F32, AdamWConfig::F32, GemmOperands::ExactF32),
        (
            dir_untied(),
            Precision::Bf16,
            AdamWConfig {
                update: UpdateRule::bf16_default(9),
                moments: MomentStorage::F32,
            },
            GemmOperands::Bf16,
        ),
    ] {
        let (rt_a, a) = load_dir(&dir, precision);
        let (rt_b, b) = load_dir(&dir, precision);
        rt_a.set_serialized_dispatch(true).unwrap();
        let table = a.parameter_table().unwrap();
        let wd = distinct_decay(&table);
        let (mut sa, mut sb) = (
            AdamW::with_config(&a, config).unwrap(),
            AdamW::with_config(&b, config).unwrap(),
        );
        let mut ids = ids_in(&dir);
        for step in 1..=3 {
            ids.rotate_left(1);
            let la = a
                .train_step_fused_adamw(&ids, mm, Supervise::Causal, &mut sa, &hyper(), &wd)
                .unwrap();
            let lb = {
                let _scope = rt_b.async_scope();
                let l = b
                    .train_step_fused_adamw(&ids, mm, Supervise::Causal, &mut sb, &hyper(), &wd)
                    .unwrap();
                rt_b.synchronize().unwrap();
                l
            };
            assert_eq!(la.to_bits(), lb.to_bits(), "{} step {step}: loss", dir.display());
            let (want, got) = (state_bits(&rt_a, &a, &table, &sa), state_bits(&rt_b, &b, &table, &sb));
            assert!(
                want == got,
                "{} {config} step {step}: async differs from serialized",
                dir.display()
            );
        }
    }
}
