//! The Qwen3.5 tower on MLX affine 4-bit weights (`Qwen35Model::load_mlx_q4`):
//! loaded straight from the converter's safetensors into the Q4 GEMV / GEMM
//! banks, with no bf16 intermediate.
//!
//! Two kinds of test, both on the device (the lead runs them under the machine
//! lock; none runs on hosted CI):
//!
//! * a synthetic checkpoint written here in the exact layout `mlx_lm.convert`
//!   writes (`language_model.model.*` keys, `U32` packed words, `[rows, cols /
//!   64]` scales and biases, `conv1d.weight` as `[conv_dim, k, 1]`, norms
//!   stored as `1 + w`), next to the same weights dequantized into a dense F32
//!   checkpoint. The Q4 model must reproduce the dense one to bf16-activation
//!   noise, which pins nibble order, scale/bias interleave, the rows-concatenated
//!   projection windows, the norm and conv conventions and the head, with no
//!   dependency on mlx-lm. Its refusals and the typed training refusal are here
//!   too.
//! * the real 2B: `mlx_q4_2b_matches_mlx_lm_logits` against mlx-lm's own logits
//!   on the same quantized weights, and a decode smoke test, both ignored (they
//!   need `tools/qwen35_ref/make_mlx_q4.py`'s output).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::logit_dev::{load_ref_in, logit_dev, LogitDev};
use common::with_gpu;
use tessl::qwen35::{AttnProjLayout, GdnProjLayout};
use tessl::qwen35_model::{LayerKind, Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_q4::{MlxQ4Error, NormConvention};
use tessl::safetensors::{CheckpointView, SafeTensors};
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};
use tessl::GpuRuntime;

// The one peak-footprint reader (also the bench bins' and the other model tests').
#[path = "../src/bin/common/footprint.rs"]
mod footprint;

const QUANT_JSON: &str =
    r#"{"quantization": {"group_size": 64, "bits": 4}, "quantization_config": {"group_size": 64, "bits": 4}}"#;

// ------------------------------------------------ a synthetic MLX checkpoint ---

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`, rounded to a bf16 value (what a bf16 model's
    /// scales, biases and norms are).
    fn bf16(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        bf16_bits_to_f32(f32_to_bf16_bits(lo + u * (hi - lo)))
    }

    fn nibble(&mut self) -> u8 {
        (self.next() >> 60) as u8
    }
}

/// One quantized matrix: 4-bit values and per-group (scale, bias), group 64.
struct QMat {
    rows: usize,
    cols: usize,
    q: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

impl QMat {
    fn new(rng: &mut Rng, rows: usize, cols: usize) -> Self {
        let groups = rows * (cols / 64);
        Self {
            rows,
            cols,
            q: (0..rows * cols).map(|_| rng.nibble()).collect(),
            scales: (0..groups).map(|_| rng.bf16(0.004, 0.02)).collect(),
            biases: (0..groups).map(|_| rng.bf16(-0.08, 0.0)).collect(),
        }
    }

    /// `w = scale * q + bias`, row-major `[rows, cols]`.
    fn dequant(&self) -> Vec<f32> {
        let g = self.cols / 64;
        (0..self.rows * self.cols)
            .map(|i| {
                let (r, c) = (i / self.cols, i % self.cols);
                self.scales[r * g + c / 64] * f32::from(self.q[i]) + self.biases[r * g + c / 64]
            })
            .collect()
    }

    /// MLX's packing: eight values per `U32`, element `i` at bits `4i..4i+4`.
    fn words(&self) -> Vec<u32> {
        self.q
            .chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |w, (i, &n)| w | (u32::from(n) << (4 * i)))
            })
            .collect()
    }
}

/// A safetensors entry: name, dtype, shape, little-endian bytes.
type Entry = (String, &'static str, Vec<usize>, Vec<u8>);

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| f32_to_bf16_bits(x).to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn safetensors(entries: &[Entry]) -> SafeTensors {
    let (mut header, mut data) = (String::from("{"), Vec::new());
    for (i, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        header += &format!(
            "{}\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            if i > 0 { "," } else { "" },
            shape.join(","),
            data.len(),
            data.len() + bytes.len()
        );
        data.extend_from_slice(bytes);
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header.into_bytes());
    out.extend(data);
    SafeTensors::from_bytes("qwen35_mlx_q4 test", out).unwrap()
}

/// Four layers (GDN, attention, GDN, attention) at the tiny fixture's widths
/// (hidden 64, MLP 128, vocabulary 64), two GDN value heads per key head, and
/// an untied head: every projection's width is a multiple of the group size,
/// and the head path is the untied one.
fn config() -> Qwen35Config {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train/config.json");
    let tiny = Qwen35Config::from_config_file(&fixture).unwrap();
    Qwen35Config {
        layers: vec![
            LayerKind::LinearAttention,
            LayerKind::FullAttention,
            LayerKind::LinearAttention,
            LayerKind::FullAttention,
        ],
        gdn: GdnProjLayout::new(1, 2, 128).unwrap(),
        attn: AttnProjLayout::new(2, 1, 256).unwrap(),
        tie_word_embeddings: false,
        ..tiny
    }
}

/// The weights of the synthetic model, once, from which both checkpoints are
/// written.
struct Net {
    cfg: Qwen35Config,
    /// (name under the tower's prefix, matrix), in the loader's read order is
    /// not required: names are all that matters.
    mats: Vec<(String, QMat)>,
    /// (name under the prefix, `w`): the zero-centred norms, as transformers
    /// stores them.
    norms: Vec<(String, Vec<f32>)>,
    /// (name under the prefix, `[conv_dim, k]` values).
    convs: Vec<(String, Vec<f32>)>,
    /// (name, values): `A_log`, `dt_bias` and the GDN gated norm.
    plain: Vec<(String, Vec<f32>)>,
    head: QMat,
}

impl Net {
    fn new(seed: u64) -> Self {
        let cfg = config();
        let mut rng = Rng(seed);
        let (h, inter) = (cfg.hidden as usize, cfg.intermediate as usize);
        let (g, a) = (cfg.gdn, cfg.attn);
        let mut mats = Vec::new();
        let (mut norms, mut convs, mut plain) = (Vec::new(), Vec::new(), Vec::new());
        mats.push((
            "embed_tokens.weight".to_string(),
            QMat::new(&mut rng, cfg.vocab as usize, h),
        ));
        for (l, kind) in cfg.layers.iter().enumerate() {
            let p = |s: &str| format!("layers.{l}.{s}");
            match kind {
                LayerKind::LinearAttention => {
                    let pw = g.part_widths();
                    for (part, w) in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"].iter().zip(pw) {
                        mats.push((p(&format!("linear_attn.{part}.weight")), QMat::new(&mut rng, w, h)));
                    }
                    mats.push((
                        p("linear_attn.out_proj.weight"),
                        QMat::new(&mut rng, h, g.value_dim() as usize),
                    ));
                    let conv_dim = g.conv_dim() as usize;
                    let k = cfg.conv_kernel as usize;
                    convs.push((
                        p("linear_attn.conv1d.weight"),
                        (0..conv_dim * k).map(|_| rng.bf16(-0.4, 0.4)).collect(),
                    ));
                    let vh = g.v_heads() as usize;
                    plain.push((p("linear_attn.A_log"), (0..vh).map(|_| rng.bf16(-0.5, 0.5)).collect()));
                    plain.push((p("linear_attn.dt_bias"), (0..vh).map(|_| rng.bf16(-0.5, 0.5)).collect()));
                    plain.push((
                        p("linear_attn.norm.weight"),
                        (0..g.v_dim() as usize).map(|_| rng.bf16(0.5, 1.5)).collect(),
                    ));
                }
                LayerKind::FullAttention => {
                    let pw = a.part_widths();
                    for (part, w) in ["q_proj", "k_proj", "v_proj"].iter().zip(pw) {
                        mats.push((p(&format!("self_attn.{part}.weight")), QMat::new(&mut rng, w, h)));
                    }
                    mats.push((
                        p("self_attn.o_proj.weight"),
                        QMat::new(&mut rng, h, (a.q_heads() * a.head_dim()) as usize),
                    ));
                    let d = a.head_dim() as usize;
                    norms.push((
                        p("self_attn.q_norm.weight"),
                        (0..d).map(|_| rng.bf16(-0.3, 0.3)).collect(),
                    ));
                    norms.push((
                        p("self_attn.k_norm.weight"),
                        (0..d).map(|_| rng.bf16(-0.3, 0.3)).collect(),
                    ));
                }
            }
            for part in ["gate_proj", "up_proj"] {
                mats.push((p(&format!("mlp.{part}.weight")), QMat::new(&mut rng, inter, h)));
            }
            mats.push((p("mlp.down_proj.weight"), QMat::new(&mut rng, h, inter)));
            norms.push((
                p("input_layernorm.weight"),
                (0..h).map(|_| rng.bf16(-0.3, 0.3)).collect(),
            ));
            norms.push((
                p("post_attention_layernorm.weight"),
                (0..h).map(|_| rng.bf16(-0.3, 0.3)).collect(),
            ));
        }
        norms.push(("norm.weight".to_string(), (0..h).map(|_| rng.bf16(-0.3, 0.3)).collect()));
        let head = QMat::new(&mut rng, cfg.vocab as usize, h);
        Self {
            cfg,
            mats,
            norms,
            convs,
            plain,
            head,
        }
    }

    /// What `mlx_lm.convert -q --q-bits 4 --q-group-size 64` writes for this
    /// model: `language_model.model.*`, the head at `language_model.lm_head`,
    /// packed words with bf16 scales and biases, norms stored `1 + w`, the
    /// convolution as `[conv_dim, k, 1]`.
    fn mlx_entries(&self) -> Vec<Entry> {
        let pre = "language_model.model.";
        let mut e: Vec<Entry> = Vec::new();
        let mut push_q = |name: String, m: &QMat| {
            let base = name.strip_suffix(".weight").unwrap().to_string();
            let words: Vec<u8> = m.words().iter().flat_map(|w| w.to_le_bytes()).collect();
            e.push((name, "U32", vec![m.rows, m.cols / 8], words));
            for (suffix, v) in [("scales", &m.scales), ("biases", &m.biases)] {
                e.push((
                    format!("{base}.{suffix}"),
                    "BF16",
                    vec![m.rows, m.cols / 64],
                    bf16_bytes(v),
                ));
            }
        };
        for (n, m) in &self.mats {
            push_q(format!("{pre}{n}"), m);
        }
        push_q("language_model.lm_head.weight".to_string(), &self.head);
        let k = self.cfg.conv_kernel as usize;
        for (n, v) in &self.convs {
            e.push((format!("{pre}{n}"), "BF16", vec![v.len() / k, k, 1], bf16_bytes(v)));
        }
        for (n, v) in &self.norms {
            let shifted: Vec<f32> = v.iter().map(|w| 1.0 + w).collect();
            e.push((format!("{pre}{n}"), "BF16", vec![v.len()], bf16_bytes(&shifted)));
        }
        for (n, v) in &self.plain {
            let (dtype, bytes) = if n.ends_with("A_log") {
                ("F32", f32_bytes(v))
            } else {
                ("BF16", bf16_bytes(v))
            };
            e.push((format!("{pre}{n}"), dtype, vec![v.len()], bytes));
        }
        e
    }

    /// The same model as a dense F32 checkpoint under `model.`, each quantized
    /// matrix dequantized exactly, norms as transformers stores them.
    fn dense_entries(&self) -> Vec<Entry> {
        let mut e: Vec<Entry> = Vec::new();
        for (n, m) in &self.mats {
            e.push((
                format!("model.{n}"),
                "F32",
                vec![m.rows, m.cols],
                f32_bytes(&m.dequant()),
            ));
        }
        e.push((
            "lm_head.weight".to_string(),
            "F32",
            vec![self.head.rows, self.head.cols],
            f32_bytes(&self.head.dequant()),
        ));
        let k = self.cfg.conv_kernel as usize;
        for (n, v) in &self.convs {
            e.push((format!("model.{n}"), "F32", vec![v.len() / k, 1, k], f32_bytes(v)));
        }
        for (n, v) in self.norms.iter().chain(&self.plain) {
            e.push((format!("model.{n}"), "F32", vec![v.len()], f32_bytes(v)));
        }
        e
    }
}

fn rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0.0f64, |m, &w| m.max(f64::from(w).abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((f64::from(g) - f64::from(w)).abs()));
    err / peak.max(f64::MIN_POSITIVE)
}

fn seq(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 64).collect()
}

fn load_q4(rt: &Arc<GpuRuntime>, entries: &[Entry]) -> Result<Qwen35Model, String> {
    let st = safetensors(entries);
    Qwen35Model::load_mlx_q4(rt, &st, config(), QUANT_JSON).map(|(m, _)| m)
}

/// Bound on the relative max-abs logit difference between the Q4 model (bf16
/// activations, the weights exactly as quantized) and the F32 model on the
/// dequantized weights: the same weights, so only the bf16 rounding of each
/// GEMM's input differs. The tiny Bf16-vs-F32 forward is held to 2e-2
/// (`tests/qwen35_model.rs`); 3e-2 leaves room for the Q4 head's bf16 input
/// too. A wrong nibble, scale, window offset or norm convention is O(1).
const TINY_BOUND: f64 = 3e-2;

/// The synthetic checkpoint's Q4 forward reproduces the dequantized dense
/// forward, at one row (decode GEMV is not in play for forward, but the head's
/// and the projections' small-M kernel is), five rows (the 2..=8 GEMM) and
/// seventy (the TensorOps GEMM, past the GDN chunk).
#[test]
fn mlx_q4_tiny_checkpoint_matches_its_dequantized_f32_model() {
    with_gpu(|rt| {
        let net = Net::new(7);
        let report_st = safetensors(&net.mlx_entries());
        let (q4, report) = Qwen35Model::load_mlx_q4(rt, &report_st, config(), QUANT_JSON).unwrap();
        assert!(q4.is_mlx_q4());
        assert_eq!(report.convention, NormConvention::MlxShifted);
        assert_eq!(report.names.prefix, "language_model.model.");
        assert_eq!(report.names.head, "language_model.lm_head");
        assert!(report.skipped.mtp.is_empty() && report.skipped.vision == 0);
        assert!(report.q4_bytes > 0, "no quantized bank was counted");
        assert!(report.peak_footprint_after >= report.peak_footprint_before && report.peak_footprint_after > 0);
        eprintln!("tiny load: {}", q4.describe());

        let dense_st = safetensors(&net.dense_entries());
        let dense = Qwen35Model::load(rt, &dense_st, "model.", config(), Precision::F32).unwrap();
        let mut worst = 0.0f64;
        for t in [1usize, 5, 70] {
            let ids = seq(t);
            let got = q4.forward(&ids, false).unwrap().logits;
            let want = dense.forward(&ids, false).unwrap().logits;
            assert!(got.iter().all(|x| x.is_finite()), "t={t}: non-finite Q4 logits");
            let r = rel(&got, &want);
            worst = worst.max(r);
            assert!(
                r <= TINY_BOUND,
                "t={t}: Q4 vs dequantized F32: rel err {r:.3e} > {TINY_BOUND:.0e}"
            );
        }
        eprintln!("tiny Q4 vs dequantized F32: worst rel err {worst:.2e} (bound {TINY_BOUND:.0e})");
    });
}

/// Decode smoke: the Q4 path's prefill then decoded tokens match its own
/// forward at the last row, in the style of
/// `prefill_then_decode_matches_forward_at_the_last_row`. Prefill lengths cover
/// the small-M GEMM (1, 5, 8) and the TensorOps GEMM (9, 70); every decoded
/// step runs the GEMV. Bound 2e-2, the tiny Bf16 model's.
#[test]
fn mlx_q4_tiny_prefill_then_decode_matches_forward_at_the_last_row() {
    const STEPS: u32 = 3;
    const BOUND: f64 = 2e-2;
    with_gpu(|rt| {
        let net = Net::new(11);
        let model = load_q4(rt, &net.mlx_entries()).unwrap();
        let mut worst = 0.0f64;
        for n in [1usize, 5, 8, 9, 70] {
            let ids = seq(n + STEPS as usize);
            let (mut dec, first) = model.prefill(&ids[..n], STEPS).unwrap();
            let mut got = vec![first];
            for k in 0..STEPS as usize {
                got.push(dec.step(ids[n + k]).unwrap());
            }
            for (k, row) in got.iter().enumerate() {
                let want = model
                    .forward_rows(&ids[..n + k], tessl::qwen35_model::LogitRows::Last, false)
                    .unwrap()
                    .logits;
                let r = rel(row, &want);
                worst = worst.max(r);
                assert!(r <= BOUND, "prefill {n} + {k} decoded: rel err {r:.3e} > {BOUND:.0e}");
            }
        }
        eprintln!("tiny Q4 decode vs forward: worst rel err {worst:.2e} (bound {BOUND:.0e})");
    });
}

/// Every unsupported layout is refused before a wrong model could run, and the
/// refusals name what they refused.
#[test]
fn mlx_q4_load_refusals() {
    with_gpu(|rt| {
        let net = Net::new(3);
        let ok = net.mlx_entries();
        let load = |entries: &[Entry], quant: &str| {
            let st = safetensors(entries);
            Qwen35Model::load_mlx_q4(rt, &st, config(), quant).map(|(_, r)| r)
        };
        let with = |extra: Entry| {
            let mut e = ok.clone_entries();
            e.push(extra);
            e
        };

        // The unmodified checkpoint loads.
        load(&ok, QUANT_JSON).unwrap();

        // A tensor under the prefix the forward never reads: refused by name
        // (a bias would be silently ignored otherwise).
        let e = load(
            &with((
                "language_model.model.layers.0.mlp.gate_proj.bias".into(),
                "BF16",
                vec![128],
                bf16_bytes(&[0.0; 128]),
            )),
            QUANT_JSON,
        )
        .unwrap_err();
        assert!(e.contains("layers.0.mlp.gate_proj.bias"), "{e}");

        // `mtp.*` outside the prefix: skipped, by name, in the report.
        let r = load(
            &with(("mtp.fc.weight".into(), "BF16", vec![2], bf16_bytes(&[0.0; 2]))),
            QUANT_JSON,
        )
        .unwrap();
        assert_eq!(r.skipped.mtp, vec!["mtp.fc.weight".to_string()]);
        // Any other tensor outside the tower: refused.
        let e = load(
            &with(("stray.weight".into(), "BF16", vec![2], bf16_bytes(&[0.0; 2]))),
            QUANT_JSON,
        )
        .unwrap_err();
        assert!(e.contains("stray.weight"), "{e}");

        // Another bit width, group size or mode, and a mixed config.
        for (quant, needle) in [
            (
                r#"{"quantization": {"group_size": 64, "bits": 8}}"#,
                "quantization.bits is 8",
            ),
            (
                r#"{"quantization": {"group_size": 32, "bits": 4}}"#,
                "quantization.group_size is 32",
            ),
            (
                r#"{"quantization": {"group_size": 64, "bits": 4, "mode": "mxfp4"}}"#,
                "mxfp4",
            ),
            (
                r#"{"quantization": {"group_size": 64, "bits": 4, "language_model.model.embed_tokens": false}}"#,
                "embed_tokens",
            ),
            (r#"{"hidden_size": 64}"#, "not an MLX-quantized"),
        ] {
            let e = load(&ok, quant).unwrap_err();
            assert!(e.contains(needle), "{e:?} lacks {needle:?}");
        }

        // A projection the converter left dense, a missing `.biases`, and an
        // 8-bit-shaped weight.
        let dense_gate = |e: &Entry| e.0 == "language_model.model.layers.0.mlp.gate_proj.weight";
        let mut dense = ok.clone_entries();
        for e in dense.iter_mut().filter(|e| dense_gate(e)) {
            *e = (e.0.clone(), "BF16", vec![128, 64], bf16_bytes(&vec![0.0; 128 * 64]));
        }
        let e = load(&dense, QUANT_JSON).unwrap_err();
        assert!(
            e.contains("layers.0.mlp.gate_proj.weight") && e.contains("mixed"),
            "{e}"
        );

        let no_biases: Vec<Entry> = ok
            .clone_entries()
            .into_iter()
            .filter(|e| e.0 != "language_model.model.layers.0.mlp.up_proj.biases")
            .collect();
        let e = load(&no_biases, QUANT_JSON).unwrap_err();
        assert!(e.contains("up_proj.weight") && e.contains("biases"), "{e}");

        let mut eight = ok.clone_entries();
        for e in eight
            .iter_mut()
            .filter(|e| e.0 == "language_model.model.layers.0.mlp.down_proj.weight")
        {
            // down_proj is [64, 128]: 8-bit would pack four values per word.
            *e = (e.0.clone(), "U32", vec![64, 32], vec![0u8; 64 * 32 * 4]);
        }
        let e = load(&eight, QUANT_JSON).unwrap_err();
        assert!(e.contains("down_proj.weight") && e.contains("other bits"), "{e}");

        // Mixed conv layouts across layers: neither convention can be assumed.
        let mut mixed = ok.clone_entries();
        for e in mixed
            .iter_mut()
            .filter(|e| e.0 == "language_model.model.layers.2.linear_attn.conv1d.weight")
        {
            e.2 = vec![512, 1, 4];
        }
        let e = load(&mixed, QUANT_JSON).unwrap_err();
        assert!(e.contains("different layout"), "{e}");

        // Not a Qwen3.5 tower at all: the drafter's unprefixed names.
        let e = load(
            &[("fc.weight".to_string(), "BF16", vec![1], bf16_bytes(&[0.0]))],
            QUANT_JSON,
        )
        .unwrap_err();
        assert_eq!(e, MlxQ4Error::NoTower.to_string());
    });
}

/// A quantized model has no dense tensors: training, the parameter table and
/// the optimizer refuse it with a message, and the answer scorer (which reads a
/// dense head) refuses too; none panics.
#[test]
fn mlx_q4_model_refuses_what_needs_dense_weights() {
    use tessl::gemm::GemmOperands;
    use tessl::qwen35_train::Qwen35Grads;
    with_gpu(|rt| {
        let net = Net::new(5);
        let model = load_q4(rt, &net.mlx_entries()).unwrap();
        let ids = seq(6);
        assert!(Qwen35Grads::zeros_like(&model).err().unwrap().contains("MLX 4-bit"));
        assert!(model.parameter_table().err().unwrap().contains("MLX 4-bit"));
        assert!(model
            .train_step(&ids, GemmOperands::Bf16)
            .map(|_| ())
            .unwrap_err()
            .contains("MLX 4-bit"));
        assert_eq!(model.train_step_bytes(6, GemmOperands::Bf16), u64::MAX);
        let staged = model.begin(&ids).unwrap();
        assert!(staged.score_answers(&[5], &[1, 2]).err().unwrap().contains("MLX 4-bit"));
        assert!(model.describe().contains("MLX affine 4-bit"), "{}", model.describe());
    });
}

/// A `Vec<Entry>` copy (the entries own their bytes).
trait CloneEntries {
    fn clone_entries(&self) -> Vec<Entry>;
}

impl CloneEntries for Vec<Entry> {
    fn clone_entries(&self) -> Vec<Entry> {
        self.iter()
            .map(|(n, d, s, b)| (n.clone(), *d, s.clone(), b.clone()))
            .collect()
    }
}

// ----------------------------------------------------------- the real 2B ---

fn q4_dir() -> PathBuf {
    std::env::var_os("QWEN35_2B_MLX_Q4_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/qwen35_2b_mlx_q4"))
}

fn load_real(rt: &Arc<GpuRuntime>) -> (Qwen35Model, tessl::qwen35_q4::MlxQ4LoadReport) {
    let dir = q4_dir();
    let cfg_text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\n(run tools/qwen35_ref/make_mlx_q4.py first, or set QWEN35_2B_MLX_Q4_DIR)",
            dir.display()
        )
    });
    let cfg = Qwen35Config::from_config_json(&cfg_text).unwrap();
    let st = CheckpointView::open(&dir).unwrap();
    Qwen35Model::load_mlx_q4(rt, &st, cfg, &cfg_text).unwrap()
}

fn ref_ids(dir: &Path) -> Vec<u32> {
    let (_, ids) = load_ref_in(dir, "ids");
    ids.iter()
        .map(|&x| u32::try_from(x as i64).expect("token id fits u32"))
        .collect()
}

/// Real Qwen3.5-2B-Base converted by `mlx_lm.convert -q --q-bits 4
/// --q-group-size 64`, loaded by tessl straight from that safetensors, against
/// mlx-lm's own logits on the same files: the default prompt (155 tokens) and
/// the natural-text prompt (512), `tools/qwen35_ref/make_mlx_q4.py`'s `ref/`.
///
/// # Bounds, fixed before the first run
///
/// Both sides compute from the identical 4-bit weights, so the quantization
/// error cancels; what is left is each side's activation precision. mlx-lm runs
/// bf16 end to end (residual stream included); tessl keeps the residual stream
/// and every accumulation in f32 and rounds only each GEMM's input to bf16.
/// `docs/qwen35.md` measures tessl's bf16 forward 18x closer to fp32 than
/// transformers' own bf16 forward, so tessl here is the near-exact side and the
/// distance to mlx-lm is mostly mlx-lm's bf16 noise: the transformers
/// bf16-vs-fp32 floor, 1e-2 of the largest logit, 152/155 top-1 (3 misses) and
/// mean KL 5.7e-4 on the default prompt (3.7e-4 and 5 misses of 512 on the
/// text). The bounds are a small multiple of that floor; a mis-read weight,
/// scale, bias, norm or conv layout puts the logits far outside them (the
/// synthetic test above shows O(1) errors for those).
///
/// * every logit finite on both sides;
/// * max |dlogit| / max |logit| <= 5e-2 (5x the floor's 1e-2);
/// * top-1 misses <= ceil(0.03 * T): 5 of 155, 16 of 512 (the floor's 3 and 5,
///   plus the a margin for a second noisy side);
/// * mean per-position KL(mlx-lm || tessl) <= 2e-3 (3.5x the floor's 5.7e-4).
///
/// Also asserted: the host peak footprint over the whole load stays under 3 GiB
/// (the bf16 2B load peaked at 5.86 GB; with no dense intermediate the 4-bit
/// load holds the ~1.4 GB of banks plus the runtime), and is recorded in the
/// printed line.
#[test]
#[ignore]
fn mlx_q4_2b_matches_mlx_lm_logits() {
    const MAX_REL: f64 = 5e-2;
    const MISS_FRACTION: f64 = 0.03;
    const MEAN_KL: f64 = 2e-3;
    const PEAK_FOOTPRINT: u64 = 3 << 30;
    with_gpu(|rt| {
        let (model, report) = load_real(rt);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "load: {}\n  quantized banks {:.2} GiB, host peak footprint {:.2} GiB after (before: {:.2} GiB), \
             convention {:?}, names {:?}, skipped mtp {} vision {}",
            model.describe(),
            gib(report.q4_bytes),
            gib(report.peak_footprint_after),
            gib(report.peak_footprint_before),
            report.convention,
            report.names,
            report.skipped.mtp.len(),
            report.skipped.vision
        );
        assert!(report.peak_footprint_after > 0, "the peak footprint was not recorded");
        assert!(
            report.peak_footprint_after <= PEAK_FOOTPRINT,
            "host peak footprint {:.2} GiB over the {:.2} GiB bound",
            gib(report.peak_footprint_after),
            gib(PEAK_FOOTPRINT)
        );
        let vocab = model.config().vocab as usize;
        let mut failures = Vec::new();
        for name in ["default", "text"] {
            let dir = q4_dir().join("ref").join(name);
            let ids = ref_ids(&dir);
            let (shape, want) = load_ref_in(&dir, "logits_mlx");
            assert_eq!(
                shape,
                vec![ids.len(), vocab],
                "{name}: reference logits are [positions, vocab]"
            );
            let got = model.forward(&ids, false).unwrap().logits;
            let d: LogitDev = logit_dev(&got, &want, vocab);
            eprintln!("{}", d.row(&format!("Q4 vs mlx-lm ({name}, {} positions)", ids.len())));
            let max_misses = (MISS_FRACTION * ids.len() as f64).ceil() as usize;
            if !d.is_finite() {
                failures.push(format!(
                    "{name}: non-finite logits ({} got, {} want)",
                    d.nonfinite_got, d.nonfinite_want
                ));
            }
            if !d.within_rel(MAX_REL) {
                failures.push(format!(
                    "{name}: max logit deviation {:.3e} of the largest logit > {MAX_REL:.0e}",
                    d.max_rel()
                ));
            }
            if d.top1_misses() > max_misses {
                failures.push(format!(
                    "{name}: top-1 misses {} of {} > {max_misses}",
                    d.top1_misses(),
                    d.positions
                ));
            }
            if !d.mean_kl.is_finite() || d.mean_kl > MEAN_KL {
                failures.push(format!("{name}: mean KL {:.3e} > {MEAN_KL:.0e}", d.mean_kl));
            }
        }
        assert!(failures.is_empty(), "Q4 vs mlx-lm:\n{}", failures.join("\n"));
    });
}

/// The decode smoke test on the real 2B's 4-bit weights: prefill `n` tokens of
/// the default prompt, decode three more, and each returned row must equal the
/// Q4 forward's last row at that length to 2e-2 of the largest logit (the tiny
/// Bf16 model's bound), with the greedy token the same unless the forward's own
/// top two logits are within that tolerance (a tie the two summation orders may
/// break either way). `n` of 5 prefills on the small-M GEMM, 150 on the
/// TensorOps one; the decoded rows run the GEMV.
#[test]
#[ignore]
fn mlx_q4_2b_prefill_then_decode_matches_forward_at_the_last_row() {
    const STEPS: usize = 3;
    const BOUND: f64 = 2e-2;
    with_gpu(|rt| {
        let (model, _) = load_real(rt);
        let ids = ref_ids(&q4_dir().join("ref").join("default"));
        for n in [5usize, 150] {
            assert!(ids.len() >= n + STEPS);
            let (mut dec, first) = model.prefill(&ids[..n], STEPS as u32).unwrap();
            let mut rows = vec![first];
            for k in 0..STEPS {
                rows.push(dec.step(ids[n + k]).unwrap());
            }
            for (k, row) in rows.iter().enumerate() {
                let want = model
                    .forward_rows(&ids[..n + k], tessl::qwen35_model::LogitRows::Last, false)
                    .unwrap()
                    .logits;
                let r = rel(row, &want);
                assert!(r <= BOUND, "prefill {n} + {k} decoded: rel err {r:.3e} > {BOUND:.0e}");
                let top = |v: &[f32]| {
                    v.iter()
                        .enumerate()
                        .fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
                };
                let (a, b) = (top(row), top(&want));
                if a.0 != b.0 {
                    let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
                    let gap = f64::from(b.1 - want[a.0]);
                    assert!(
                        gap <= BOUND * scale,
                        "prefill {n} + {k}: decoded token {} vs forward token {} with a gap of {gap:.3e}",
                        a.0,
                        b.0
                    );
                }
            }
            eprintln!("2B Q4 prefill {n} + {STEPS} decoded tokens match forward's last rows");
        }
    });
}
