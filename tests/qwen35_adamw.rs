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
use tessl::qwen35_adamw::{excluded_from_weight_decay, AdamW, AdamWHyper, Moment};
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
