//! `Qwen35Model`'s parameter table (`tessl::qwen35_params`) on the tiny
//! training fixture: the values read back are the checkpoint's under
//! transformers' names, the gradients are transformers' autograd's, a write
//! round-trips and keeps the training step and the inference forward on the
//! same model, and a bad tensor anywhere stops a write before any of it.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_params::ParamInfo;
use tessl::safetensors::SafeTensors;
use tessl::{GpuRuntime, Tensor};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn load(precision: Precision) -> (Arc<GpuRuntime>, Qwen35Model, SafeTensors) {
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let model = Qwen35Model::load(&rt, &st, "model.", cfg, precision).unwrap();
    (rt, model, st)
}

fn ids() -> Vec<u32> {
    let a = read_npy(&fixture().join("ids.npy")).unwrap();
    a.i64_slice().unwrap().iter().map(|&x| x as u32).collect()
}

fn alloc(rt: &Arc<GpuRuntime>, table: &[ParamInfo]) -> Vec<Tensor> {
    table
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect()
}

/// A caller tensor's values in transformers' layout.
fn hf(p: &ParamInfo, t: &Tensor) -> Vec<f32> {
    let d = t.read_f32().unwrap();
    if !p.transposed {
        return d;
    }
    let (out, inn) = (p.shape[0], p.shape[1]);
    (0..out)
        .flat_map(|o| (0..inn).map(move |i| (o, i)))
        .map(|(o, i)| d[i * out + o])
        .collect()
}

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

#[test]
fn values_are_the_checkpoints_under_transformers_names() {
    let (rt, model, st) = load(Precision::F32);
    let table = model.parameter_table().unwrap();
    // Every checkpoint tensor appears once (the tied head has no entry).
    let mut names: Vec<String> = table.iter().map(|p| format!("model.{}", p.name)).collect();
    let mut want: Vec<String> = st.names().map(str::to_string).collect();
    names.sort();
    want.sort();
    assert_eq!(names, want);
    let dst = alloc(&rt, &table);
    model.read_parameters(&dst).unwrap();
    for (p, t) in table.iter().zip(&dst) {
        let (shape, w) = st.read_f32(&format!("model.{}", p.name)).unwrap();
        assert_eq!(shape, p.shape, "{}", p.name);
        let got = hf(p, t);
        // The checkpoint's bits, the zero-centred norms' w included.
        for (i, (a, b)) in got.iter().zip(&w).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{}[{i}]: {a} vs {b}", p.name);
        }
    }
}

#[test]
fn gradients_are_transformers_autograd_under_its_names() {
    let (rt, model, _st) = load(Precision::F32);
    let table = model.parameter_table().unwrap();
    let step = model.train_step(&ids(), GemmOperands::ExactF32).unwrap();
    let dst = alloc(&rt, &table);
    model.read_gradients(&step.grads, &dst).unwrap();
    let mut worst = 0.0f64;
    for (p, t) in table.iter().zip(&dst) {
        let a = read_npy(&fixture().join(format!("grad.model.{}.npy", p.name))).unwrap();
        assert_eq!(a.shape, p.shape, "{}", p.name);
        let want = a.f32_slice().unwrap();
        let got = hf(p, t);
        let peak = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
        let r = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / peak;
        worst = worst.max(f64::from(r));
        assert!(r <= 1e-4, "{}: {r:.3e}", p.name);
    }
    eprintln!("worst gradient through the table: {worst:.2e}");
}

#[test]
fn a_write_round_trips_and_moves_both_forwards_together() {
    let (rt, model, _st) = load(Precision::F32);
    let ids = ids();
    let table = model.parameter_table().unwrap();
    let before = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    let params = alloc(&rt, &table);
    model.read_parameters(&params).unwrap();

    // Writing back what was read changes nothing.
    model.write_parameters(&params).unwrap();
    let same = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    assert_eq!(
        same.loss.to_bits(),
        before.loss.to_bits(),
        "a write of the read values moved the loss"
    );

    // One SGD step: p - lr * g, written back, then read again.
    let grads = alloc(&rt, &table);
    model.read_gradients(&before.grads, &grads).unwrap();
    let lr = 0.05f32;
    let stepped: Vec<Tensor> = params
        .iter()
        .zip(&grads)
        .map(|(p, g)| {
            let v: Vec<f32> = p
                .read_f32()
                .unwrap()
                .iter()
                .zip(g.read_f32().unwrap())
                .map(|(a, b)| a - lr * b)
                .collect();
            common::tensor_f32(&rt, p.shape(), &v)
        })
        .collect();
    model.write_parameters(&stepped).unwrap();
    let back = alloc(&rt, &table);
    model.read_parameters(&back).unwrap();
    for ((p, s), b) in table.iter().zip(&stepped).zip(&back) {
        let (s, b) = (s.read_f32().unwrap(), b.read_f32().unwrap());
        // Exact, the embedding (an f32 table) and the norms included.
        for (i, (x, y)) in b.iter().zip(&s).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{}[{i}]: read back {x}, wrote {y}", p.name);
        }
    }
    // Loss decreases along the negative gradient, and the training step and
    // the inference forward (whose head reads the same table) agree.
    let after = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    assert!(after.loss < before.loss, "loss {} -> {}", before.loss, after.loss);
    let infer = inference_loss(&model, &ids);
    assert!(
        (infer - after.loss).abs() <= 1e-5 * after.loss,
        "inference {infer} vs training {}",
        after.loss
    );
}

/// Any f32 written reads back with its bits, for every entry: the
/// zero-centred norms are stored as their `w`, so a `w` below -0.5 (whose
/// `1 + w` has finer steps than `w` itself) survives a write and a read, as
/// a checkpoint restored into a fresh model needs.
#[test]
fn every_written_value_reads_back_bit_for_bit() {
    let (rt, model, _st) = load(Precision::F32);
    let table = model.parameter_table().unwrap();
    // Values across [-2, 2] with full low bits, and the norm value a resumed
    // AdamW run lost an ulp of.
    let values = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                if i == 0 {
                    return -0.659_430_6;
                }
                let u = ((i + 7 * salt) as f64 * 0.618_033_988_749_894_9).fract();
                (u * 4.0 - 2.0) as f32 * (1.0 + 1e-7 * (i % 13) as f32)
            })
            .collect()
    };
    let src: Vec<Tensor> = table
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let shape = p.storage_shape();
            common::tensor_f32(&rt, &shape, &values(shape.iter().product(), k))
        })
        .collect();
    model.write_parameters(&src).unwrap();
    let back = alloc(&rt, &table);
    model.read_parameters(&back).unwrap();
    for ((p, s), b) in table.iter().zip(&src).zip(&back) {
        let (s, b) = (s.read_f32().unwrap(), b.read_f32().unwrap());
        let bad: Vec<(usize, f32, f32)> = s
            .iter()
            .zip(&b)
            .enumerate()
            .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
            .map(|(i, (x, y))| (i, *x, *y))
            .take(3)
            .collect();
        assert!(bad.is_empty(), "{}: wrote, read back {bad:?}", p.name);
    }
}

#[test]
fn copies_honour_the_callers_byte_offset() {
    let (rt, model, _st) = load(Precision::F32);
    let table = model.parameter_table().unwrap();
    let plain = alloc(&rt, &table);
    model.read_parameters(&plain).unwrap();
    // Each tensor 3 floats into a buffer poisoned with NaN.
    let offset: Vec<Tensor> = table
        .iter()
        .map(|p| {
            let n: usize = p.shape.iter().product();
            let b = rt.alloc_buffer((n + 6) * 4).unwrap();
            b.write_f32(&vec![f32::NAN; n + 6]);
            Tensor::from_buffer(&rt, b, &p.storage_shape(), tessl::DType::F32, 12).unwrap()
        })
        .collect();
    model.read_parameters(&offset).unwrap();
    for ((p, a), b) in table.iter().zip(&plain).zip(&offset) {
        let all = b.buffer.read_f32();
        let n: usize = p.shape.iter().product();
        assert!(
            all[..3].iter().chain(&all[3 + n..]).all(|x| x.is_nan()),
            "{}: wrote outside its window",
            p.name
        );
        assert_eq!(a.read_f32().unwrap(), b.read_f32().unwrap(), "{}", p.name);
    }
    // And a write from offset tensors is a write of their windows.
    model.write_parameters(&offset).unwrap();
    let again = alloc(&rt, &table);
    model.read_parameters(&again).unwrap();
    for ((p, a), b) in table.iter().zip(&plain).zip(&again) {
        assert_eq!(a.read_f32().unwrap(), b.read_f32().unwrap(), "{}", p.name);
    }
}

#[test]
fn refusals_leave_the_model_untouched() {
    let (rt, model, _st) = load(Precision::F32);
    let ids = ids();
    let table = model.parameter_table().unwrap();
    let before = model.train_step(&ids, GemmOperands::ExactF32).unwrap().loss;
    let e = |r: Result<(), String>, needle: &str| {
        let m = r.expect_err(needle);
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    let mut ts = alloc(&rt, &table);
    for t in &ts {
        let n = t.numel();
        t.buffer.write_f32(&vec![0.25; n]);
    }
    e(
        model.write_parameters(&ts[1..]),
        &format!("{} tensors for {} parameters", ts.len() - 1, ts.len()),
    );
    // The last tensor has the wrong shape: nothing before it is written.
    let last = ts.len() - 1;
    ts[last] = rt.alloc_tensor_f32(&[1]).unwrap();
    e(model.write_parameters(&ts), "norm.weight must be f32");
    let bf = common::tensor_bf16(
        &rt,
        &table[0].storage_shape(),
        &vec![0.0; table[0].shape.iter().product()],
    );
    let mut wrong = alloc(&rt, &table);
    wrong[0] = bf;
    e(model.read_parameters(&wrong), "embed_tokens.weight must be f32");
    assert_eq!(
        model.train_step(&ids, GemmOperands::ExactF32).unwrap().loss.to_bits(),
        before.to_bits()
    );

    let (_, bf16, _) = load(Precision::Bf16);
    e(bf16.read_parameters(&alloc(&rt, &table)), "Precision::F32");
}
