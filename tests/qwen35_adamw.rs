//! `Qwen35Model::adamw_step` against torch.optim.AdamW's formula in f64.
//!
//! The reference is the single-tensor path of `torch.optim.AdamW` (amsgrad
//! and maximize off) evaluated on the host in f64, fed each step with the
//! parameters and gradients tessl itself holds (read through the parameter
//! table, so the norms stored as `1 + w` are compared as `w`). Its moments
//! are its own, carried in f64 across steps.
//!
//! Bound, set before the first run: `2e-6` absolute per element, `2^-22` more
//! for the `1 + w` norms (storing `1 + w` rounds `w` at ulp(1) = 2^-23 per
//! write). At lr = 1e-2 an f32 update is good to ~1e-9, the parameter's own
//! rounding to ~6e-8 near 1, while a semantic error is 1e-5 or more (decay
//! after the moments: lr^2 * wd; a missing bias correction at step 1: 0.1 * lr;
//! eps outside the division by sqrt(bc2) at eps = 1e-2: ~0.3 * lr).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::qwen35_adamw::{excluded_from_weight_decay, AdamW, AdamWHyper};
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

fn is_one_plus(name: &str) -> bool {
    name.ends_with("layernorm.weight") || name == "norm.weight"
}

/// `steps` steps of train_step + adamw_step from a fresh state, each checked
/// against the f64 reference; returns the worst error seen.
fn run(hyper: AdamWHyper, wd_all: f32, steps: usize) -> f64 {
    let (rt, model) = load();
    let table = model.parameter_table().unwrap();
    let wd = model.default_weight_decay(wd_all).unwrap();
    let mut state = AdamW::new(&model).unwrap();
    let ids = ids();
    let mut m: Vec<Vec<f64>> = table.iter().map(|p| vec![0.0; p.shape.iter().product()]).collect();
    let mut v = m.clone();
    let mut worst = 0.0f64;
    for step in 1..=steps {
        let s = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
        let p0 = host(&rt, &model, &table, None);
        let g = host(&rt, &model, &table, Some(&s.grads));
        model.adamw_step(&s.grads, &mut state, &hyper, &wd).unwrap();
        assert_eq!(state.step_count(), step as u64);
        let p1 = host(&rt, &model, &table, None);

        let t = step as f64;
        let (bc1, bc2) = (1.0 - hyper.beta1.powf(t), 1.0 - hyper.beta2.powf(t));
        for (i, info) in table.iter().enumerate() {
            let mut moved = false;
            for k in 0..p0[i].len() {
                let mut w = p0[i][k] * (1.0 - hyper.lr * f64::from(wd[i]));
                m[i][k] += (1.0 - hyper.beta1) * (g[i][k] - m[i][k]);
                v[i][k] = v[i][k] * hyper.beta2 + (1.0 - hyper.beta2) * g[i][k] * g[i][k];
                let denom = v[i][k].sqrt() / bc2.sqrt() + hyper.eps;
                w -= hyper.lr / bc1 * m[i][k] / denom;
                let err = (p1[i][k] - w).abs();
                let bound = 2e-6 + if is_one_plus(&info.name) { 2f64.powi(-22) } else { 0.0 };
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
    worst
}

#[test]
fn steps_match_torch_adamw_in_f64() {
    let worst = run(
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
    let worst = run(
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
