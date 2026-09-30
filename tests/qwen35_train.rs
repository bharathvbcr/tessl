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

use tessl::npy::read_npy;
use tessl::qwen35_model::{LayerKind, Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{MixerGrads, Qwen35Grads, TrainStep};
use tessl::safetensors::SafeTensors;
use tessl::tensor::{GpuBuffer, Tensor};
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
    (s[0], s[1], read(&t.buffer, s[0] * s[1]))
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
    let (h, i) = (cfg.hidden as usize, cfg.intermediate as usize);
    let mut out = vec![
        (format!("{prefix}embed_tokens.weight"), read_t(&g.embed).2),
        (format!("{prefix}norm.weight"), read(&g.final_norm, h)),
    ];
    for (l, lg) in g.layers.iter().enumerate() {
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
                out.push((p("linear_attn.in_proj_z.weight"), packed(&m.w_in, gl.z_off() as usize, vd)));
                out.push((p("linear_attn.in_proj_b.weight"), packed(&m.w_in, gl.b_off() as usize, hv)));
                out.push((p("linear_attn.in_proj_a.weight"), packed(&m.w_in, gl.a_off() as usize, hv)));
                out.push((p("linear_attn.out_proj.weight"), packed(&m.w_out, 0, h)));
                out.push((p("linear_attn.conv1d.weight"), read(&m.conv_w, cd * cfg.conv_kernel as usize)));
                out.push((p("linear_attn.A_log"), read(&m.a_log, hv)));
                out.push((p("linear_attn.dt_bias"), read(&m.dt_bias, hv)));
                out.push((p("linear_attn.norm.weight"), read(&m.norm_w, gl.v_dim() as usize)));
            }
            MixerGrads::Attn(m) => {
                let al = cfg.attn;
                let (q2, kv) = ((2 * al.q_heads() * al.head_dim()) as usize, (al.kv_heads() * al.head_dim()) as usize);
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
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    Qwen35Model::load(&rt, &st, prefix, cfg, precision).unwrap()
}

fn tiny_config() -> Qwen35Config {
    Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap()
}

/// Loss and every gradient against the fixture; returns the worst relative
/// error.
fn compare(dir: &Path, prefix: &str, cfg: &Qwen35Config, step: &TrainStep, bound: f64) -> f64 {
    let (_, want_loss) = npy_f64(&dir.join("loss.npy"));
    let loss_rel = (step.loss - want_loss[0]).abs() / want_loss[0].abs();
    eprintln!("loss {:.8} vs {:.8} (rel {loss_rel:.2e})", step.loss, want_loss[0]);
    assert!(loss_rel <= 1e-5, "loss {} vs transformers {}", step.loss, want_loss[0]);
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
    let files = std::fs::read_dir(dir).unwrap().filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().starts_with("grad.")).count();
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
    let step = model.train_step(&ids).unwrap();
    let worst = compare(&dir, "model.", &cfg, &step, 1e-4);
    eprintln!("worst parameter gradient: {worst:.2e}");

    // The training forward is the inference forward: the same loss from
    // the inference logits.
    let logits = model.forward(&ids, false).unwrap().logits;
    let v = cfg.vocab as usize;
    let mut ce = 0.0f64;
    for (t, row) in logits.chunks(v).take(ids.len() - 1).enumerate() {
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let z: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
        ce += m + z.ln() - row[ids[t + 1] as usize] as f64;
    }
    ce /= (ids.len() - 1) as f64;
    assert!((ce - step.loss).abs() <= 1e-5 * ce.abs(), "inference loss {ce} vs training loss {}", step.loss);

    // A second step gives the same bits.
    let again = model.train_step(&ids).unwrap();
    assert_eq!(again.loss.to_bits(), step.loss.to_bits(), "loss changed on a rerun");
    for ((name, a), (_, b)) in by_name(&cfg, &again.grads, "model.").iter().zip(by_name(&cfg, &step.grads, "model.")) {
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{name} changed on a rerun");
    }
}

#[test]
fn train_step_refuses_what_it_does_not_implement() {
    let dir = fixture();
    let e = |r: Result<TrainStep, String>, needle: &str| {
        let m = r.err().unwrap_or_else(|| panic!("{needle}: accepted"));
        assert!(m.contains(needle), "{m:?} lacks {needle:?}");
    };
    let bf16 = load(&dir, "model.", tiny_config(), Precision::Bf16);
    e(bf16.train_step(&[1, 2, 3]), "training runs in f32");
    let model = load(&dir, "model.", tiny_config(), Precision::F32);
    e(model.train_step(&[5]), "at least two tokens");
    e(model.train_step(&[5, 64]), "token id 64 >= vocab 64");
}

/// Qwen3.5-2B-Base: `python3 tools/qwen35_ref/make_train_fixture.py 2b`
/// first (see its docstring for memory), then this. Compares the loss, every
/// 1-D and conv parameter, layers 0 and 3 in full, and the embedding rows
/// the reference kept.
#[test]
#[ignore]
fn real_2b_step_matches_transformers() {
    let dir = std::env::var_os("QWEN35_TRAIN_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/qwen35_train_ref"));
    let st_path = PathBuf::from(
        std::env::var("QWEN35_2B_SAFETENSORS").expect("set QWEN35_2B_SAFETENSORS to the Qwen3.5-2B-Base .safetensors file"),
    );
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&st_path).unwrap();
    let cfg = Qwen35Config::qwen35_2b().unwrap();
    let model = Qwen35Model::load(&rt, &st, "model.language_model.", cfg.clone(), Precision::F32).unwrap();
    drop(st);
    let step = model.train_step(&ids(&dir)).unwrap();
    // The embedding: only the rows the reference kept.
    let (_, rows) = npy_f64(&dir.join("embed_rows.npy"));
    let (_, want) = npy_f64(&dir.join("grad.model.language_model.embed_tokens.weight.rows.npy"));
    let (_, h, all) = read_t(&step.grads.embed);
    let got: Vec<f64> = rows.iter().flat_map(|&r| all[r as usize * h..][..h].to_vec()).collect();
    let r = rel(&got, &want);
    eprintln!("embedding rows: {r:.2e}");
    assert!(r <= 1e-3, "embedding rows rel err {r:.3e}");
    let worst = compare_subset(&dir, "model.language_model.", &cfg, &step);
    eprintln!("worst parameter gradient: {worst:.2e}");
}

/// [`compare`] for a reference that kept a subset (and the embedding as rows).
fn compare_subset(dir: &Path, prefix: &str, cfg: &Qwen35Config, step: &TrainStep) -> f64 {
    let (_, want_loss) = npy_f64(&dir.join("loss.npy"));
    let loss_rel = (step.loss - want_loss[0]).abs() / want_loss[0].abs();
    eprintln!("loss {:.8} vs {:.8} (rel {loss_rel:.2e})", step.loss, want_loss[0]);
    assert!(loss_rel <= 1e-5, "loss {} vs transformers {}", step.loss, want_loss[0]);
    let mut worst = 0.0f64;
    let mut seen = 1; // the embedding rows, compared by the caller
    for (name, got) in by_name(cfg, &step.grads, prefix) {
        let path = dir.join(format!("grad.{name}.npy"));
        if !path.exists() {
            continue;
        }
        let r = rel(&got, &npy_f64(&path).1);
        eprintln!("{name}: {r:.2e}");
        assert!(r <= 1e-3, "{name}: rel err {r:.3e} > 1e-3");
        worst = worst.max(r);
        seen += 1;
    }
    let files = std::fs::read_dir(dir).unwrap().filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().starts_with("grad.")).count();
    assert_eq!(seen, files, "every reference gradient must be compared");
    worst
}
