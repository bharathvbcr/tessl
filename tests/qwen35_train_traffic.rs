//! What a Qwen3.5 training step costs the host besides its kernels: host
//! zeroing, waits on the GPU, and the bits of its gradients when the work
//! moves around. `tests/fixtures/qwen35_train/` is the small random model
//! `tests/qwen35_train.rs` checks against transformers.
//!
//! The counters ([`tessl::infer_trace`]) are process-wide, so every test here
//! holds [`LOCK`] while it runs.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tessl::cross_entropy::{cross_entropy_rows, CeGrads, CeHidden, CeWorkspace, Reduction};
use tessl::gemm::GemmOperands;
use tessl::infer_trace;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{MixerGrads, Qwen35Grads, Supervise};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{DType, GpuBuffer, Tensor};
use tessl::GpuRuntime;

static LOCK: Mutex<()> = Mutex::new(());

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn ids() -> Vec<u32> {
    let a = tessl::npy::read_npy(&fixture().join("ids.npy")).unwrap();
    a.i64_slice().unwrap().iter().map(|&x| x as u32).collect()
}

fn load(precision: Precision) -> (Arc<GpuRuntime>, Qwen35Model) {
    let rt = GpuRuntime::new().unwrap();
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let model = match precision {
        Precision::F32 => Qwen35Model::load(&rt, &st, "model.", cfg, precision).unwrap(),
        Precision::Bf16 => Qwen35Model::load_tower(&rt, &st, "model.", cfg, precision).unwrap(),
    };
    (rt, model)
}

/// What `f` counted.
fn traced<T>(f: impl FnOnce() -> T) -> (T, infer_trace::Snapshot) {
    infer_trace::set_enabled(true);
    let s0 = infer_trace::snapshot();
    let out = f();
    let s1 = infer_trace::snapshot();
    infer_trace::set_enabled(false);
    (out, s1.since(&s0))
}

/// Every gradient's bits, in a fixed order, and the loss's.
fn bits(loss: f64, g: &Qwen35Grads) -> Vec<(String, Vec<u32>)> {
    fn t(x: &Tensor) -> Vec<u32> {
        let n = x.numel();
        match x.dtype {
            DType::BF16 => x.buffer.contents_u16()[x.byte_offset() / 2..][..n]
                .iter()
                .map(|&b| u32::from(b))
                .collect(),
            _ => x.buffer.read_f32()[x.byte_offset() / 4..][..n]
                .iter()
                .map(|v| v.to_bits())
                .collect(),
        }
    }
    fn b(x: &GpuBuffer) -> Vec<u32> {
        x.read_f32().iter().map(|v| v.to_bits()).collect()
    }
    let mut v = vec![
        (
            "loss".to_string(),
            vec![loss.to_bits() as u32, (loss.to_bits() >> 32) as u32],
        ),
        ("embed".into(), t(&g.embed)),
        ("final_norm".into(), b(&g.final_norm)),
    ];
    for (i, l) in g.layers.iter().enumerate() {
        let mut add = |name: &str, x: Vec<u32>| v.push((format!("layers.{i}.{name}"), x));
        add("input_norm", b(&l.input_norm));
        add("post_norm", b(&l.post_norm));
        match &l.mixer {
            MixerGrads::Gdn(m) => {
                add("w_in", t(&m.w_in));
                add("w_out", t(&m.w_out));
                add("conv_w", b(&m.conv_w));
                add("a_log", b(&m.a_log));
                add("dt_bias", b(&m.dt_bias));
                add("norm_w", b(&m.norm_w));
            }
            MixerGrads::Attn(m) => {
                add("w_in", t(&m.w_in));
                add("w_out", t(&m.w_out));
                add("q_norm", b(&m.q_norm));
                add("k_norm", b(&m.k_norm));
            }
        }
        add("gate", t(&l.gate));
        add("up", t(&l.up));
        add("down", t(&l.down));
    }
    v
}

/// The step's entry points on one model, each into its own gradients: a
/// fresh `train_step` (an f32 model only), a causal sequence into a bank, a
/// `Supervise::Rows` step added to it, and a span step (no rows scored in
/// tessl, a gradient handed back at three positions) added to that.
fn entry_points(rt: &Arc<GpuRuntime>, model: &Qwen35Model, op: GemmOperands) -> Vec<(String, Vec<u32>)> {
    let ids = ids();
    let (t, h) = (ids.len(), model.config().hidden as usize);
    let mut out = Vec::new();
    let mut take = |tag: &str, loss: f64, g: &Qwen35Grads| {
        out.extend(bits(loss, g).into_iter().map(|(n, v)| (format!("{tag}: {n}"), v)));
    };
    if model.precision() == Precision::F32 {
        let s = model.train_step(&ids, op).unwrap();
        take("train_step", s.loss, &s.grads);
    }
    let bank = Qwen35Grads::zeros_like(model).unwrap();
    let l = model
        .train_step_into(&ids, op, Supervise::Causal, &bank, false)
        .unwrap();
    take("bank, causal", l, &bank);
    let pos: Vec<u32> = [1, t / 2, t - 2].iter().map(|&p| p as u32).collect();
    let tgt: Vec<u32> = pos.iter().map(|&p| ids[p as usize + 1]).collect();
    let rev: Vec<u32> = ids.iter().rev().copied().collect();
    let sup = Supervise::Rows {
        positions: &pos,
        targets: &tgt,
        scale: 0.25,
    };
    let l = model.train_step_into(&rev, op, sup, &bank, true).unwrap();
    take("bank, + rows", l, &bank);
    let none = Supervise::Rows {
        positions: &[],
        targets: &[],
        scale: 1.0,
    };
    let p = model.train_forward(&ids, op, none).unwrap();
    let hidden = rt.alloc_tensor_f32(&[pos.len(), h]).unwrap();
    p.hidden(&pos, &hidden).unwrap();
    let dh = rt.alloc_tensor_f32(&[pos.len(), h]).unwrap();
    let scaled: Vec<f32> = hidden.buffer.read_f32()[..pos.len() * h]
        .iter()
        .map(|x| x * 1e-2)
        .collect();
    dh.buffer.write_f32(&scaled);
    model.train_backward_into(p, Some((&pos, &dh)), &bank, true).unwrap();
    take("bank, + span", 0.0, &bank);
    out
}

/// The three models and operand lanes the step runs on.
fn lanes() -> Vec<(&'static str, Precision, GemmOperands)> {
    vec![
        ("f32 model, exact f32", Precision::F32, GemmOperands::ExactF32),
        ("f32 model, bf16 operands", Precision::F32, GemmOperands::Bf16),
        ("bf16 model", Precision::Bf16, GemmOperands::Bf16),
    ]
}

/// A step's outputs, temporaries and workspaces are all written by a kernel
/// before anything reads them, so none is zeroed on the host: at T = 2048 on
/// the 2B that was 808 allocations and 25.9 GB per step, a CPU pass with
/// the GPU idle. What still zeroes on the host is pinned here: a gradient
/// bank (one hot tensor per weight matrix; its vectors are zeroed through
/// their mapping) and the parallel split-K TN's partition scratch, whose
/// `mode::multiply` partitions accumulate into it.
#[test]
fn a_step_zeroes_nothing_on_the_host() {
    let _g = LOCK.lock().unwrap();
    for (lane, precision, op) in lanes() {
        let (rt, model) = load(precision);
        let (_, s) = traced(|| entry_points(&rt, &model, op));
        // The entry points' own allocations are a bank (one per matrix)
        // and two f32 tensors for the span step's rows.
        let matrices = 1 + 5 * model.config().layers.len() as u64;
        assert_eq!(
            s.host_zeros,
            matrices + 2,
            "{lane}: {} host-zeroed allocations ({} B) beyond the bank's {matrices} and the test's 2",
            s.host_zeros.saturating_sub(matrices + 2),
            s.host_zero_bytes
        );
    }
    let (rt, _) = load(Precision::F32);
    let (a, b, c) = (
        rt.alloc_tensor_f32(&[4096, 12]).unwrap(),
        rt.alloc_tensor_f32(&[4096, 768]).unwrap(),
        rt.alloc_tensor_f32(&[12, 768]).unwrap(),
    );
    let (_, s) = traced(|| GemmOperands::ExactF32.tn(&a, &b, &c).unwrap());
    assert_eq!(s.host_zeros, 1, "the split-K TN's partition scratch is zeroed once");
}

/// Every allocation the step leaves unzeroed, filled with NaN instead
/// (`set_poison_unzeroed`), changes no bit of any gradient or loss on any
/// entry point: nothing reads a byte of them before a kernel writes it.
#[test]
fn poisoned_unzeroed_allocations_change_no_bit() {
    let _g = LOCK.lock().unwrap();
    for (lane, precision, op) in lanes() {
        let (rt, model) = load(precision);
        let clean = entry_points(&rt, &model, op);
        rt.set_poison_unzeroed(true);
        let poisoned = entry_points(&rt, &model, op);
        rt.set_poison_unzeroed(false);
        assert_eq!(clean.len(), poisoned.len());
        for ((name, a), (_, b)) in clean.iter().zip(&poisoned) {
            if let Some(k) = (0..a.len()).find(|&k| a[k] != b[k]) {
                panic!(
                    "{lane}: {name}[{k}] is {:#x} clean, {:#x} with unzeroed allocations poisoned",
                    a[k], b[k]
                );
            }
        }
    }
}

/// Under async encode a step waits for the GPU only where it means to: at
/// each attention layer's forward and rebuild (where the pool recycles what
/// earlier layers freed, which `train_step_bytes` counts on), at the
/// cross-entropy's loss, and at its end. Index uploads (the attention
/// workspace's, the cross-entropy's rows and targets, the embedding
/// backward's grouping), `dxf`'s zero and the backward's scratch used to
/// add a drain each; the attention workspace was also rebuilt per layer.
#[test]
fn an_async_step_waits_only_where_it_means_to() {
    let _g = LOCK.lock().unwrap();
    let (rt, model) = load(Precision::F32);
    let ids = ids();
    let attn = model
        .config()
        .layers
        .iter()
        .filter(|&&k| k == tessl::qwen35_model::LayerKind::FullAttention)
        .count() as u64;
    assert!(attn > 0, "the fixture has no attention layer to count");
    rt.set_async_encode(true).unwrap();
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    for op in [GemmOperands::ExactF32, GemmOperands::Bf16] {
        let (_, fresh) = traced(|| model.train_step(&ids, op).unwrap());
        let (_, into) = traced(|| model.train_step_into(&ids, op, Supervise::Causal, &bank, true).unwrap());
        for (what, s) in [("train_step", fresh), ("train_step_into", into)] {
            assert_eq!(s.sync_waits, 2 * attn + 2, "{op:?} {what}: {s:?}");
        }
    }
    rt.set_async_encode(false).unwrap();
}

/// Async encode changes when the step waits, not what it computes: every
/// gradient bit and the loss on every entry point equal the sync run's.
#[test]
fn async_encode_changes_no_bit() {
    let _g = LOCK.lock().unwrap();
    for (lane, precision, op) in lanes() {
        let (rt, model) = load(precision);
        let sync = entry_points(&rt, &model, op);
        rt.set_async_encode(true).unwrap();
        let batched = entry_points(&rt, &model, op);
        rt.set_async_encode(false).unwrap();
        for ((name, a), (_, b)) in sync.iter().zip(&batched) {
            if let Some(k) = (0..a.len()).find(|&k| a[k] != b[k]) {
                panic!("{lane}: {name}[{k}] is {:#x} sync, {:#x} async", a[k], b[k]);
            }
        }
    }
}

/// A step into an f32 bank that it overwrites writes every gradient in
/// place: the weight matrices and the embedding (head and gather) by the
/// GEMMs and the gather themselves, and every vector gradient by its own
/// kernel (each overwrites its output). Beyond the fresh step it dispatches
/// nothing. Each matrix used to be a fresh tensor and a copy, each vector a
/// fresh buffer and a copy, and the head a fresh `[vocab, hidden]` tensor
/// (2 GB on the 2B, 4 GiB at 4B) zeroed, filled and then added.
#[test]
fn an_f32_bank_takes_its_gradients_in_place() {
    let _g = LOCK.lock().unwrap();
    let (_rt, model) = load(Precision::F32);
    let ids = ids();
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    for op in [GemmOperands::ExactF32, GemmOperands::Bf16] {
        let (_, fresh) = traced(|| model.train_step(&ids, op).unwrap());
        let (_, into) = traced(|| {
            model
                .train_step_into(&ids, op, Supervise::Causal, &bank, false)
                .unwrap()
        });
        assert_eq!(
            into.dispatches,
            fresh.dispatches,
            "{op:?}: a step into the bank dispatched {} beyond the fresh step's {}",
            into.dispatches.saturating_sub(fresh.dispatches),
            fresh.dispatches
        );
    }
}

const CHILD: &str = "TESSL_QWEN35_TRAIN_TRAFFIC_CHILD";

/// Run test `name` again in a child process with the kernel trace on (it
/// latches at first use, so the parent cannot switch it), alone.
fn in_traced_child(name: &str, body: impl FnOnce()) {
    if std::env::var_os(CHILD).is_some() {
        body();
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "--test-threads=1", name])
        .env(CHILD, "1")
        .env("TESSL_KERNEL_TRACE", "1")
        .output()
        .expect("spawn the traced child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{name} failed in its child:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("1 passed"), "the child ran no test:\n{stdout}");
}

/// Every sum the step forms from two products is formed in the second's
/// GEMM: the forward residual (`resid + y @ w_out`), the MLP backward's
/// `d_gate @ gate^T + d_up @ up^T`, and the cross-entropy's `dh` over the
/// vocabulary's chunks. None is a product into a temporary plus a
/// `qwen35_residual_add_f32` pass any more, on either operand lane.
#[test]
fn the_step_adds_its_products_in_the_gemms() {
    in_traced_child("the_step_adds_its_products_in_the_gemms", || {
        let (rt, model) = load(Precision::F32);
        let ids = ids();
        let (n, h, v) = (5usize, model.config().hidden as usize, model.config().vocab as usize);
        for op in [GemmOperands::ExactF32, GemmOperands::Bf16] {
            model.train_step(&ids, op).unwrap();
            // Several vocabulary chunks (the step's own covers the fixture's
            // 64 tokens in one).
            let hid = rt.alloc_tensor_f32(&[n, h]).unwrap();
            hid.buffer
                .write_f32(&(0..n * h).map(|i| (i % 7) as f32 * 0.1).collect::<Vec<_>>());
            let w = rt.alloc_tensor_f32(&[v, h]).unwrap();
            w.buffer
                .write_f32(&(0..v * h).map(|i| (i % 5) as f32 * 0.01).collect::<Vec<_>>());
            let ws = CeWorkspace::new(&rt, n as u32, h as u32, 16, DType::F32, op).unwrap();
            let (dh, dw) = (
                rt.alloc_tensor_f32(&[n, h]).unwrap(),
                rt.alloc_tensor_f32(&[v, h]).unwrap(),
            );
            let rows: Vec<u32> = (0..n as u32).collect();
            let targets: Vec<u32> = (0..n as u32).map(|i| i * 11 % v as u32).collect();
            let grads = CeGrads {
                dh: &dh,
                dw: &dw,
                scale: 1.0,
            };
            let hidden = CeHidden { rows: &hid, off: 0 };
            cross_entropy_rows(&rt, hidden, &w, &rows, &targets, Reduction::Mean, op, &ws, Some(grads)).unwrap();
        }
        rt.synchronize().unwrap();
        let ran = tessl::runtime::traced_kernels();
        assert!(!ran.is_empty(), "the kernel trace recorded nothing");
        assert!(
            !ran.iter().any(|k| k == "qwen35_residual_add_f32"),
            "a separate residual add ran: {ran:?}"
        );
        for k in [
            "matmul2d_tensorops_nn_accum_f32",
            "matmul2d_tensorops_nt_accum_f32",
            "matmul2d_tensorops_nt_accum_bf16_f32",
        ] {
            assert!(ran.iter().any(|r| r == k), "{k} never ran: {ran:?}");
        }
        assert!(
            ran.iter().any(|r| r.starts_with("matmul2d_tensorops_bf16_f32_epi")),
            "the bf16 accumulate (the epilogue, beta = 1) never ran: {ran:?}"
        );
    });
}

/// On a bf16-stored model's lane (a bf16 weight, bf16 operands) the
/// cross-entropy reads the weight as it is and rounds each f32 operand once:
/// no `cast_bf16_to_f32` at all, and `cast_f32_to_bf16` once for the
/// gathered rows plus once per vocabulary chunk for its dlogits (once in
/// all without gradients). It used to widen each weight chunk to f32 in both
/// walks for the GEMMs to round it back, and to round the rows for every
/// GEMM and each dlogits chunk for two. In a whole bf16-storage step the
/// only widening left is each layer's kept input, once, for its rebuild.
#[test]
fn a_bf16_storage_cross_entropy_casts_each_operand_once() {
    in_traced_child("a_bf16_storage_cross_entropy_casts_each_operand_once", || {
        let count = tessl::runtime::traced_kernel_count;
        let (rt, model) = load(Precision::Bf16);
        let op = GemmOperands::Bf16;
        let (n, h, v, chunk) = (5usize, model.config().hidden as usize, 300usize, 64u32);
        let hid = rt.alloc_tensor_f32(&[n + 2, h]).unwrap();
        hid.buffer
            .write_f32(&(0..(n + 2) * h).map(|i| (i % 7) as f32 * 0.1).collect::<Vec<_>>());
        let w = rt.alloc_tensor_bf16(&[v, h]).unwrap();
        w.buffer.write_bf16_bits(
            &(0..v * h)
                .map(|i| ((((i % 5) as f32 - 2.0) * 0.01).to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
        );
        let ws = CeWorkspace::new(&rt, n as u32, h as u32, chunk, DType::BF16, op).unwrap();
        let (dh, dw) = (
            rt.alloc_tensor_f32(&[n, h]).unwrap(),
            rt.alloc_tensor_f32(&[v, h]).unwrap(),
        );
        let rows: Vec<u32> = vec![0, 2, 4, 6, 1];
        let targets: Vec<u32> = vec![0, 63, 64, 299, 150];
        let chunks = v.div_ceil(chunk as usize) as u64;
        for with_grads in [false, true] {
            let (w0, n0) = (count("cast_bf16_to_f32"), count("cast_f32_to_bf16"));
            let grads = with_grads.then_some(CeGrads {
                dh: &dh,
                dw: &dw,
                scale: 1.0,
            });
            let hidden = CeHidden { rows: &hid, off: 0 };
            cross_entropy_rows(&rt, hidden, &w, &rows, &targets, Reduction::Mean, op, &ws, grads).unwrap();
            let (widened, rounded) = (count("cast_bf16_to_f32") - w0, count("cast_f32_to_bf16") - n0);
            assert_eq!(
                widened, 0,
                "grads {with_grads}: the cross-entropy widened {widened} times"
            );
            let want = 1 + if with_grads { chunks } else { 0 };
            assert_eq!(
                rounded, want,
                "grads {with_grads}: {rounded} roundings, want {want} (the rows once, each dlogits chunk once)"
            );
        }
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        let w0 = count("cast_bf16_to_f32");
        model
            .train_step_into(&ids(), op, Supervise::Causal, &bank, false)
            .unwrap();
        let widened = count("cast_bf16_to_f32") - w0;
        let layers = model.config().layers.len() as u64;
        assert_eq!(
            widened, layers,
            "a bf16-storage step widened {widened} times; only its {layers} kept layer inputs should be"
        );
    });
}

/// Into a bf16 bank the head's gradient goes a chunk at a time (one
/// `qwen35_head_rows_to_bank_bf16` per vocabulary chunk, one here) and the
/// step's token rows once the gather has added (one
/// `qwen35_token_rows_to_bank_bf16`), so `qwen35_deliver_f32_to_bf16` runs
/// only for the layers' five bf16 matrices each. It ran once more, for a
/// `[vocab, hidden]` f32 embedding gradient delivered whole.
#[test]
fn a_bf16_bank_takes_the_head_gradient_by_chunk() {
    in_traced_child("a_bf16_bank_takes_the_head_gradient_by_chunk", || {
        let count = tessl::runtime::traced_kernel_count;
        let (_rt, model) = load(Precision::Bf16);
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        let kernels = [
            "qwen35_deliver_f32_to_bf16",
            "qwen35_head_rows_to_bank_bf16",
            "qwen35_token_rows_to_bank_bf16",
        ];
        let before = kernels.map(count);
        model
            .train_step_into(&ids(), GemmOperands::Bf16, Supervise::Causal, &bank, false)
            .unwrap();
        let ran = kernels.map(count);
        let layers = model.config().layers.len() as u64;
        let chunks = u64::from(model.config().vocab.div_ceil(8192));
        assert_eq!(
            [ran[0] - before[0], ran[1] - before[1], ran[2] - before[2]],
            [5 * layers, chunks, 1],
            "{kernels:?}"
        );
    });
}

/// A step into a bf16 bank holds no `[vocab, hidden]` f32 head gradient:
/// its peak above what it starts with stays below that tensor's size (128
/// MiB here; 2.0 GB on the 2B, 2.5 GB at the 4B's shapes), where the head
/// gradient was a fresh f32 tensor rounded into the bank at the end. It
/// holds one `[8192, hidden]` chunk of it instead (32 MiB), and the peak
/// stays under the step's bound. The device's allocation counts every
/// runtime in the process, so this holds the lock.
#[test]
fn a_bf16_bank_step_holds_no_f32_head_gradient() {
    let _g = LOCK.lock().unwrap();
    let cfg = Qwen35Config {
        vocab: 32_768,
        hidden: 1024,
        ..Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap()
    };
    let head = cfg.vocab as u64 * cfg.hidden as u64 * 4;
    let rt = GpuRuntime::new().unwrap();
    let model = Qwen35Model::random_tower(&rt, cfg.clone(), Precision::Bf16, 5).unwrap();
    let bank = Qwen35Grads::zeros_like(&model).unwrap();
    for t in [8u32, 64] {
        let ids: Vec<u32> = (0..t).map(|i| (i * 7_919 + 3) % cfg.vocab).collect();
        rt.synchronize().unwrap();
        let before = rt.current_allocated_bytes();
        rt.reset_peak_allocated_bytes();
        model
            .train_step_into(&ids, GemmOperands::Bf16, Supervise::Causal, &bank, false)
            .unwrap();
        let peak = rt.peak_allocated_bytes().saturating_sub(before);
        let bound = model.train_step_bytes(t, GemmOperands::Bf16);
        eprintln!("T={t}: peak {peak} B, the f32 head gradient {head} B, bound {bound} B");
        assert!(
            peak < head,
            "T={t}: peak {peak} B is not below the f32 head gradient's {head} B"
        );
        assert!(peak <= bound, "T={t}: peak {peak} over the bound {bound}");
    }
}

/// Every parameter's f32 bits, as `read_parameters` lays them out.
fn params(rt: &Arc<GpuRuntime>, model: &Qwen35Model) -> Vec<u32> {
    let ts: Vec<Tensor> = model
        .parameter_table()
        .unwrap()
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect();
    model.read_parameters(&ts).unwrap();
    ts.iter()
        .flat_map(|t| t.read_f32().unwrap().into_iter().map(f32::to_bits))
        .collect()
}

/// An AdamW step over every parameter is one table-driven dispatch per step
/// kernel (one on an f32 model, where it was one per parameter) and one
/// wait; `adamw_step_unwaited` makes no wait, and the same parameters, bit
/// for bit, once something does wait.
#[test]
fn an_adamw_step_is_one_dispatch_and_waits_only_if_asked() {
    let _g = LOCK.lock().unwrap();
    let ids = ids();
    let hyper = tessl::qwen35_adamw::AdamWHyper {
        lr: 1e-3,
        grad_scale: 0.5,
        ..Default::default()
    };
    let mut after = Vec::new();
    for unwaited in [false, true] {
        let (rt, model) = load(Precision::F32);
        let wd = model.default_weight_decay(0.1).unwrap();
        let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
        let mut state = tessl::qwen35_adamw::AdamW::new(&model).unwrap();
        rt.set_async_encode(true).unwrap();
        let (_, s) = traced(|| {
            if unwaited {
                model.adamw_step_unwaited(&step.grads, &mut state, &hyper, &wd).unwrap()
            } else {
                model.adamw_step(&step.grads, &mut state, &hyper, &wd).unwrap()
            }
        });
        assert_eq!(s.dispatches, 1, "unwaited {unwaited}: {s:?}");
        assert_eq!(s.sync_waits, u64::from(!unwaited), "unwaited {unwaited}: {s:?}");
        assert_eq!(state.step_count(), 1);
        rt.set_async_encode(false).unwrap();
        after.push(params(&rt, &model));
    }
    assert!(
        after[0] == after[1],
        "the unwaited step moved the parameters differently"
    );
}

/// `grad_sq_norm` is one command buffer: one table-driven dispatch over
/// every gradient, and one that adds its pieces on the GPU. So it waits once
/// in either encode mode, reads back five words rather than a value per row,
/// and zeroes nothing on the host. It was a dispatch per parameter, each a
/// wait under sync encode (27 here, 320 on the 2B), and a read of every
/// row's sum.
#[test]
fn grad_sq_norm_is_one_command_buffer_and_one_wait() {
    let _g = LOCK.lock().unwrap();
    let ids = ids();
    for (lane, precision, op) in lanes() {
        let (rt, model) = load(precision);
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        model
            .train_step_into(&ids, op, Supervise::Causal, &bank, false)
            .unwrap();
        let mut seen = Vec::new();
        for batched in [false, true] {
            rt.set_async_encode(batched).unwrap();
            let (sq, s) = traced(|| model.grad_sq_norm(&bank).unwrap());
            rt.set_async_encode(false).unwrap();
            assert!(sq.is_finite() && sq > 0.0, "{lane}: {sq}");
            assert_eq!(
                (s.dispatches, s.commits, s.sync_waits, s.host_zeros),
                (2, 1, 1, 0),
                "{lane}, async {batched}: {s:?}"
            );
            seen.push(sq.to_bits());
        }
        assert_eq!(seen[0], seen[1], "{lane}: the encode mode changed the norm");
    }
}

/// The step count of an unwaited step counts at once, and rolls back to the
/// last waited count when the runtime is poisoned (a GPU fault leaves the
/// moments unknown); a waited step's stays.
#[test]
fn an_unwaited_step_count_rolls_back_on_a_poisoned_runtime() {
    let _g = LOCK.lock().unwrap();
    let ids = ids();
    let (rt, model) = load(Precision::F32);
    let wd = model.default_weight_decay(0.0).unwrap();
    let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
    let mut state = tessl::qwen35_adamw::AdamW::new(&model).unwrap();
    let hyper = tessl::qwen35_adamw::AdamWHyper::default();
    model.adamw_step(&step.grads, &mut state, &hyper, &wd).unwrap();
    rt.set_async_encode(true).unwrap();
    model.adamw_step_unwaited(&step.grads, &mut state, &hyper, &wd).unwrap();
    model.adamw_step_unwaited(&step.grads, &mut state, &hyper, &wd).unwrap();
    assert_eq!(state.step_count(), 3);
    rt.poison_as_shared_event_timeout_for_test();
    assert_eq!(state.step_count(), 1, "the unwaited steps were not rolled back");
}

/// The GDN kernels read q, k and v through column windows of the conv output
/// and write dq, dk and dv into the conv gradient's columns, key heads
/// shared by their value heads: a step runs no `qwen35_copy_cols_f32` (it
/// used to copy q, k, v out of the conv output and their gradients back, and
/// on grouped heads repeat each key head and sum each gradient back, per
/// layer), on the ungrouped fixture and the grouped one (4 value heads over
/// 2 key heads), on every lane. A step still runs the GDN kernels.
#[test]
fn a_gdn_step_copies_no_columns() {
    in_traced_child("a_gdn_step_copies_no_columns", || {
        let count = tessl::runtime::traced_kernel_count;
        for dir in ["qwen35_train", "qwen35_train_grouped"] {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir);
            let ids = tessl::npy::read_npy(&root.join("ids.npy")).unwrap();
            let ids: Vec<u32> = ids.i64_slice().unwrap().iter().map(|&x| x as u32).collect();
            let rt = GpuRuntime::new().unwrap();
            let cfg = Qwen35Config::from_config_file(&root.join("config.json")).unwrap();
            let st = SafeTensors::open(&root.join("model.safetensors")).unwrap();
            for (lane, precision, op) in lanes() {
                let model = match precision {
                    Precision::F32 => Qwen35Model::load(&rt, &st, "model.", cfg.clone(), precision).unwrap(),
                    Precision::Bf16 => Qwen35Model::load_tower(&rt, &st, "model.", cfg.clone(), precision).unwrap(),
                };
                // Whichever GDN backward production runs: the token-sequential kernel or the chunked one.
                let gdn_bwd = || count("gdn_train_bwd") + count("gdn_train_c_bwd_a");
                let (copies, gdn) = (count("qwen35_copy_cols_f32"), gdn_bwd());
                let bank = Qwen35Grads::zeros_like(&model).unwrap();
                model
                    .train_step_into(&ids, op, Supervise::Causal, &bank, false)
                    .unwrap();
                rt.synchronize().unwrap();
                assert!(gdn_bwd() > gdn, "{dir}, {lane}: the step ran no GDN backward");
                assert_eq!(
                    count("qwen35_copy_cols_f32"),
                    copies,
                    "{dir}, {lane}: a training step copied columns"
                );
            }
        }
    });
}
