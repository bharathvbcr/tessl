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
use tessl::safetensors::SafeTensors;
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

fn moments(rt: &Arc<GpuRuntime>, model: &Qwen35Model, table: &[ParamInfo], s: &AdamW) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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
    f64::from(tessl::tensor::bf16_bits_to_f32(tessl::tensor::f32_to_bf16_bits(x as f32)))
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
                    let want = reference(&hyper(), step, wd[i], master0[i][k], g[i][k], &mut m[i][k], &mut v[i][k]);
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
/// of `p`) plus the f32 bound: `|p1 + c1 - ref| <= 2^-9 |c1| + 2^-17 |ref|
/// + 2e-6`. Set before the first run. At lr = 1e-4 most updates are below
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
                let want = reference(&h, step, wd[i], p0[i][k] + c0[i][k], g[i][k], &mut m[i][k], &mut v[i][k]);
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
    eprintln!("kahan: worst one-step error {worst:.2e}; drift from the reference with c {with_c:.3e}, without {without:.3e}");
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
        assert!(state.describe().contains(&format!("seed={seed}")), "{}", state.describe());
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
