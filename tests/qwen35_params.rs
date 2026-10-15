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

/// The bf16 load places every projection part and the transposed LM head
/// from the same checkpoint values the f32 load reads (the fixture's weights
/// are bf16, so both hold them exactly): the two forwards differ only by bf16
/// GEMM inputs. A part placed in the wrong columns, or a head that is not the
/// embedding's transpose, moves the logits by their own size, not by that.
#[test]
fn a_bf16_load_holds_the_same_weights_as_the_f32_load() {
    let (_, f32_model, _) = load(Precision::F32);
    let (_, bf16_model, _) = load(Precision::Bf16);
    let ids = ids();
    let want = f32_model.forward(&ids, false).unwrap().logits;
    let got = bf16_model.forward(&ids, false).unwrap().logits;
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
    let worst = got.iter().zip(&want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(scale > 0.0 && worst.is_finite());
    assert!(worst / scale < 2e-2, "bf16 logits off by {worst} at scale {scale}");
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

    // Another runtime's tensors are refused before anything is copied.
    let (brt, bf16, _) = load(Precision::Bf16);
    e(
        bf16.read_parameters(&alloc(&rt, &table)),
        "belongs to a different runtime",
    );
    // A bf16 model carrying the inference head's packed copy of the
    // embedding reads (widened) but refuses a write, which would leave that
    // copy stale.
    bf16.read_parameters(&alloc(&brt, &table)).unwrap();
    e(bf16.write_parameters(&alloc(&brt, &table)), "load_tower");
}

// ---- per-entry host copies (`*_host`) ----

fn load_tower(precision: Precision) -> (Arc<GpuRuntime>, Qwen35Model) {
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let model = Qwen35Model::load_tower(&rt, &st, "model.", cfg, precision).unwrap();
    (rt, model)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every parameter on the host through the whole-table copy.
fn whole(rt: &Arc<GpuRuntime>, model: &Qwen35Model, table: &[ParamInfo]) -> Vec<Vec<f32>> {
    let ts = alloc(rt, table);
    model.read_parameters(&ts).unwrap();
    ts.iter().map(|t| t.read_f32().unwrap()).collect()
}

fn numel(p: &ParamInfo) -> usize {
    p.shape.iter().product()
}

/// Each entry, f32 and bf16 bank alike (every packed projection past the
/// first in its bank has a window offset): the per-entry read is the
/// whole-table read's bits, a gradient likewise, a write of a read changes
/// nothing, and a write of new values lands in that entry alone.
#[test]
fn per_entry_host_copies_match_the_whole_table_copies() {
    for precision in [Precision::F32, Precision::Bf16] {
        let (rt, model) = load_tower(precision);
        let table = model.parameter_table().unwrap();
        let ids = ids();
        let step = model
            .train_step(
                &ids,
                if precision == Precision::F32 {
                    GemmOperands::ExactF32
                } else {
                    GemmOperands::Bf16
                },
            )
            .unwrap();
        let params = whole(&rt, &model, &table);
        let grads = alloc(&rt, &table);
        model.read_gradients(&step.grads, &grads).unwrap();
        for (i, p) in table.iter().enumerate() {
            let mut got = vec![0.0f32; numel(p)];
            model.read_parameter_host(i, &mut got).unwrap();
            assert_eq!(bits(&got), bits(&params[i]), "{precision:?} {} parameter", p.name);
            model.read_gradient_host(&step.grads, i, &mut got).unwrap();
            assert_eq!(
                bits(&got),
                bits(&grads[i].read_f32().unwrap()),
                "{precision:?} {} gradient",
                p.name
            );
        }
        // A read written back is the identity (bf16: widen then round is exact).
        for (i, p) in table.iter().enumerate() {
            model.write_parameter_host(i, &params[i]).unwrap();
            assert_eq!(
                whole(&rt, &model, &table).iter().map(|v| bits(v)).collect::<Vec<_>>(),
                params.iter().map(|v| bits(v)).collect::<Vec<_>>(),
                "{precision:?} {}: write of a read moved something",
                p.name
            );
        }
        // New values land in their entry alone; bf16 takes them rounded to nearest.
        for (i, p) in table.iter().enumerate() {
            let new: Vec<f32> = (0..numel(p)).map(|k| 0.37 + 0.0123 * (k % 31) as f32).collect();
            model.write_parameter_host(i, &new).unwrap();
            let after = whole(&rt, &model, &table);
            let matrix = p.transposed || p.name == "embed_tokens.weight";
            let stored = |x: f32| match precision {
                Precision::Bf16 if matrix => tessl::tensor::bf16_bits_to_f32(tessl::tensor::f32_to_bf16_bits(x)),
                _ => x,
            };
            let want: Vec<f32> = new.iter().map(|&x| stored(x)).collect();
            assert_eq!(bits(&after[i]), bits(&want), "{precision:?} {} written", p.name);
            for (j, q) in table.iter().enumerate().filter(|(j, _)| *j != i) {
                assert_eq!(
                    bits(&after[j]),
                    bits(&params[j]),
                    "{precision:?} writing {} moved {}",
                    p.name,
                    q.name
                );
            }
            model.write_parameter_host(i, &params[i]).unwrap();
        }
    }
}

/// A bad index or length, a head that cannot be written, or a busy runtime is
/// a named refusal that moves nothing: values and a live `Decode` (which
/// refuses once the generation moves) are as they were.
#[test]
fn per_entry_host_refusals_move_nothing() {
    let (rt, model, _st) = load(Precision::F32);
    let table = model.parameter_table().unwrap();
    let ids = ids();
    let before = whole(&rt, &model, &table);
    let (mut dec, _) = model.prefill(&ids[..4], 2).unwrap();
    let n = numel(&table[1]);
    let e = |r: Result<(), String>, needle: &str| {
        let m = r.expect_err(needle);
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    let len = table.len();
    e(
        model.write_parameter_host(len, &[1.0]),
        &format!("parameter index {len} out of range for {len} entries"),
    );
    e(model.read_parameter_host(len, &mut [0.0]), "out of range");
    let mut buf = vec![0.0; n];
    e(model.write_parameter_host(1, &vec![9.0; n + 1]), &table[1].name);
    e(model.write_parameter_host(1, &[]), "takes");
    e(model.read_parameter_host(1, &mut buf[1..]), "takes");
    e(model.read_parameter_host(1, &mut []), "takes");
    // Busy: another mapping on the same runtime is the existing refusal.
    {
        let other = rt.alloc_tensor_f32(&[1]).unwrap();
        let _held = other.buffer.try_contents_f32().unwrap();
        e(model.read_parameter_host(1, &mut buf), "runtime busy");
        e(model.write_parameter_host(1, &vec![9.0; n]), "runtime busy");
        let g = model.train_step(&ids, GemmOperands::ExactF32);
        assert!(g.is_err(), "a train step ran under a host mapping");
    }
    let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    {
        let held = rt.alloc_tensor_f32(&[1]).unwrap();
        let _held = held.buffer.try_contents_f32().unwrap();
        e(model.read_gradient_host(&step.grads, 1, &mut buf), "runtime busy");
    }
    e(model.read_gradient_host(&step.grads, 1, &mut buf[1..]), "takes");
    // Another runtime's gradients are refused by name.
    let (_, other, _) = load(Precision::F32);
    let theirs = other.train_step(&ids, GemmOperands::ExactF32).unwrap();
    e(model.read_gradient_host(&theirs.grads, 1, &mut buf), "another runtime");
    // Nothing moved: values the same, the session still continues.
    assert_eq!(
        whole(&rt, &model, &table).iter().map(|v| bits(v)).collect::<Vec<_>>(),
        before.iter().map(|v| bits(v)).collect::<Vec<_>>()
    );
    dec.step(ids[4]).expect("a refused write bumped the generation");
    // The inference head's packed copy cannot be left stale.
    let (_, packed, _) = load(Precision::Bf16);
    e(packed.write_parameter_host(1, &vec![0.0; n]), "load_tower");
    let mut out = vec![0.0; n];
    packed.read_parameter_host(1, &mut out).unwrap();
}

// ---- adversarial: per-entry host writes against the GPU path ----

/// Awkward f32 bit patterns first (bf16 ties both ways, overflow to infinity,
/// quiet and signalling NaNs with payloads in the dropped bits, subnormals,
/// signed zeros and infinities), then xorshift noise over all of f32's bits.
fn awkward(n: usize, seed: u32) -> Vec<f32> {
    const FIXED: [u32; 24] = [
        0x3f80_8000,
        0x3f81_8000,
        0x3f80_8001,
        0x3f80_7fff,
        0xbf80_8000,
        0xbf81_8000,
        0x7f7f_8000,
        0x7f7f_ffff,
        0xff7f_8000,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fc0_0001,
        0x7fff_ffff,
        0xffff_ffff,
        0x7f80_8000,
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

/// `write_parameter_host` and the GPU `write_parameters` must store the same
/// bf16 bits for the same f32 input, whatever the input (ties, overflow, NaN
/// payloads, subnormals): a host write that rounds differently from the
/// training path would make a restored checkpoint differ from a trained one.
#[test]
fn host_write_rounds_like_the_gpu_write_bit_for_bit() {
    let (rt, model) = load_tower(Precision::Bf16);
    let table = model.parameter_table().unwrap();
    let vals: Vec<Vec<f32>> = table
        .iter()
        .enumerate()
        .map(|(i, p)| awkward(numel(p), 0x9e37_79b9 ^ i as u32))
        .collect();
    let ts = alloc(&rt, &table);
    for (t, v) in ts.iter().zip(&vals) {
        t.write_f32(v).unwrap();
    }
    model.write_parameters(&ts).unwrap();
    let gpu: Vec<Vec<u32>> = table
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut v = vec![0.0f32; numel(p)];
            model.read_parameter_host(i, &mut v).unwrap();
            bits(&v)
        })
        .collect();
    // Reset every entry to something else so a no-op host write cannot pass.
    for (i, p) in table.iter().enumerate() {
        model.write_parameter_host(i, &vec![0.5; numel(p)]).unwrap();
    }
    for (i, p) in table.iter().enumerate() {
        model.write_parameter_host(i, &vals[i]).unwrap();
        let mut v = vec![0.0f32; numel(p)];
        model.read_parameter_host(i, &mut v).unwrap();
        let host = bits(&v);
        let diff: Vec<(usize, u32, u32, u32)> = (0..host.len())
            .filter(|&k| host[k] != gpu[i][k])
            .map(|k| (k, vals[i][k].to_bits(), gpu[i][k], host[k]))
            .take(4)
            .collect();
        assert!(
            diff.is_empty(),
            "{}: (index, input bits, gpu bits, host bits) {diff:x?}",
            p.name
        );
    }
}

/// A poisoned runtime is an `Err` from every per-entry call, not a panic or a
/// silent success, and a refused write leaves the parameters unmoved.
#[test]
fn per_entry_host_calls_refuse_on_a_poisoned_runtime() {
    let (rt, model, _st) = load(Precision::F32);
    let table = model.parameter_table().unwrap();
    let ids = ids();
    let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    let n = numel(&table[1]);
    rt.poison_as_shared_event_timeout_for_test();
    let mut buf = vec![0.0f32; n];
    for (what, r) in [
        ("read_parameter_host", model.read_parameter_host(1, &mut buf)),
        ("read_gradient_host", model.read_gradient_host(&step.grads, 1, &mut buf)),
        ("write_parameter_host", model.write_parameter_host(1, &vec![7.0; n])),
    ] {
        assert!(r.is_err(), "{what} succeeded on a poisoned runtime");
    }
}
