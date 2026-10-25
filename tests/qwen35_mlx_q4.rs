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
        self.dense_entries_as(false)
    }

    /// [`Self::dense_entries`] with the matrices (the tables and projections)
    /// stored BF16, which is what a `Precision::Bf16` dense load requires
    /// (`SafeTensors::read_bf16_bits_into` accepts BF16 only, and always did);
    /// the small tensors stay as they are, since the f32 reads take either. The
    /// dequantized values round to bf16 here, which is exact when the net is
    /// `make_exact`.
    fn dense_entries_bf16(&self) -> Vec<Entry> {
        self.dense_entries_as(true)
    }

    fn dense_entries_as(&self, bf16: bool) -> Vec<Entry> {
        let matrix = |name: String, m: &QMat| -> Entry {
            let w = m.dequant();
            let (dtype, bytes) = if bf16 {
                ("BF16", bf16_bytes(&w))
            } else {
                ("F32", f32_bytes(&w))
            };
            (name, dtype, vec![m.rows, m.cols], bytes)
        };
        let mut e: Vec<Entry> = Vec::new();
        for (n, m) in &self.mats {
            e.push(matrix(format!("model.{n}"), m));
        }
        e.push(matrix("lm_head.weight".to_string(), &self.head));
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
        assert!(report.skipped.mtp.is_empty() && report.skipped.vision.is_empty());
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

/// The fused Q4 decode step (`DecodeFusion::ALL`: the gate/up GEMVs and SiLU·mul
/// in one dispatch) is bit-identical to the separate kernels, logits compared as
/// bits over every step, and saves exactly two dispatches per layer under async
/// encode (one commit, one wait either way). Add+norm changes nothing on Q4: the
/// GEMV already adds the residual in its own dispatch.
#[test]
fn mlx_q4_tiny_fused_decode_is_bit_identical_and_saves_two_dispatches_per_layer() {
    use tessl::infer_trace;
    use tessl::qwen35_model::DecodeFusion;
    const STEPS: usize = 4;
    with_gpu(|rt| {
        let model = load_q4(rt, &Net::new(13).mlx_entries()).unwrap();
        let layers = config().layers.len() as u64;
        for n in [1usize, 5, 70] {
            let ids = seq(n + STEPS);
            let run = |fusion: DecodeFusion| {
                rt.set_async_encode(true).unwrap();
                let (mut dec, first) = model.prefill(&ids[..n], STEPS as u32).unwrap();
                dec.set_fusion(fusion);
                let mut rows = vec![first];
                let mut counts = None;
                for k in 0..STEPS {
                    infer_trace::set_enabled(k == STEPS - 1);
                    let before = infer_trace::snapshot();
                    rows.push(dec.step(ids[n + k]).unwrap());
                    if k == STEPS - 1 {
                        counts = Some(infer_trace::snapshot().since(&before));
                    }
                }
                infer_trace::set_enabled(false);
                rt.set_async_encode(false).unwrap();
                (rows, counts.unwrap())
            };
            let (base, c0) = run(DecodeFusion::NONE);
            let (fused, c1) = run(DecodeFusion::ALL);
            for (k, (a, b)) in fused.iter().zip(&base).enumerate() {
                let (a, b): (Vec<u32>, Vec<u32>) = (
                    a.iter().map(|x| x.to_bits()).collect(),
                    b.iter().map(|x| x.to_bits()).collect(),
                );
                assert_eq!(
                    a, b,
                    "prefill {n} + {k}: fused Q4 step differs from the separate kernels"
                );
            }
            assert_eq!(c1.dispatches, c0.dispatches - 2 * layers, "prefill {n}");
            assert_eq!((c0.commits, c0.sync_waits, c1.commits, c1.sync_waits), (1, 1, 1, 1));
            eprintln!(
                "Q4 prefill {n}: {} dispatches per token unfused, {} fused",
                c0.dispatches, c1.dispatches
            );
        }
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
/// # The reference is mlx-lm run in FLOAT32 (owner decision, on this evidence)
///
/// The gate compares tessl with `logits_mlx_f32.npy`: mlx-lm on the same 4-bit
/// weights with every floating parameter cast to float32 (written by
/// `make_mlx_q4.py`, rewritten without reconverting by `--refs-only`). It first
/// compared with mlx-lm's bfloat16 logits, and the default prompt missed the
/// top-1 bound by one (6 misses against 5). The attribution run
/// (`mlx_q4_2b_attribution_against_mlx_lm_float32`, job 18) against the float32
/// reference settled whose rounding that was:
///
/// | | tessl Q4 | mlx-lm bf16 |
/// |---|---|---|
/// | default, 155: top-1 misses / max rel / mean KL | 0 / 3.2e-3 / 3.7e-5 | 6 / 1.28e-2 / 7.8e-4 |
/// | text, 512: top-1 misses / max rel / mean KL | 3 / 3.8e-3 / 3.0e-5 | 10 / 1.15e-2 / 4.6e-4 |
///
/// tessl is closer to the exact computation on these weights than mlx-lm's own
/// bf16 forward is, on every figure and both prompts, so the misses against
/// mlx-lm bf16 are its activation rounding. The bounds below are the ones
/// written before the first run, UNCHANGED; only the reference moved. The
/// comparison with mlx-lm bf16 is still printed, as context, and not gated.
///
/// # Bounds, fixed before the first run
///
/// Both sides compute from the identical 4-bit weights, so the quantization
/// error cancels; what is left is each side's activation precision. tessl keeps
/// the residual stream and every accumulation in f32 and rounds only each
/// GEMM's input to bf16. `docs/qwen35.md` measures tessl's bf16 forward 18x
/// closer to fp32 than transformers' own bf16 forward. The bounds were set as a
/// small multiple of the transformers bf16-vs-fp32 floor: 1e-2 of the largest
/// logit, 152/155 top-1 (3 misses) and mean KL 5.7e-4 on the default prompt
/// (3.7e-4 and 5 misses of 512 on the text). A mis-read weight, scale, bias,
/// norm or conv layout puts the logits far outside them (the synthetic test
/// above shows O(1) errors for those).
///
/// * every logit finite on both sides;
/// * max |dlogit| / max |logit| <= 5e-2 (5x the floor's 1e-2);
/// * top-1 misses <= ceil(0.03 * T): 5 of 155, 16 of 512 (the floor's 3 and 5,
///   plus the a margin for a second noisy side);
/// * mean per-position KL(mlx-lm float32 || tessl) <= 2e-3 (3.5x the floor's 5.7e-4).
///
/// Also asserted: the host peak footprint after the load stays under 3 GiB
/// (the bf16 2B load peaked at 5.86 GB; with no dense intermediate the 4-bit
/// load holds the ~1.4 GB of banks plus the runtime), and is recorded in the
/// printed line.
///
/// The footprint is the PROCESS LIFETIME peak, so it measures the load only
/// when nothing large was allocated earlier in the process. Run this test alone
/// in its own process (`-- --ignored --exact mlx_q4_2b_matches_mlx_lm_logits`,
/// as the doc's commands do); it loads first, before any other work, and
/// asserts the figure before the load was under 1 GiB and that the load raised
/// it, so a run that shares a process with a bigger earlier allocation fails
/// instead of reporting a number that is not the load's.
#[test]
#[ignore]
fn mlx_q4_2b_matches_mlx_lm_logits() {
    const MAX_REL: f64 = 5e-2;
    const MISS_FRACTION: f64 = 0.03;
    const MEAN_KL: f64 = 2e-3;
    const PEAK_FOOTPRINT: u64 = 3 << 30;
    const PEAK_BEFORE: u64 = 1 << 30;
    with_gpu(|rt| {
        let (model, report) = load_real(rt);
        assert!(
            report.peak_footprint_before <= PEAK_BEFORE && report.peak_footprint_after > report.peak_footprint_before,
            "the process lifetime peak was {} B before the load and {} B after: run this test alone in a fresh \
             process, or the figure is not the load's",
            report.peak_footprint_before,
            report.peak_footprint_after
        );
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
            report.skipped.vision.len()
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
            let f32_path = dir.join("logits_mlx_f32.npy");
            assert!(
                f32_path.exists(),
                "{}: missing. The gate compares with mlx-lm in float32; write it with \
                 `python tools/qwen35_ref/make_mlx_q4.py --refs-only` (keeps the converted checkpoint)",
                f32_path.display()
            );
            let (shape, want) = load_ref_in(&dir, "logits_mlx_f32");
            assert_eq!(
                shape,
                vec![ids.len(), vocab],
                "{name}: reference logits are [positions, vocab]"
            );
            let got = model.forward(&ids, false).unwrap().logits;
            // Context, not gated: the same forward against mlx-lm's bfloat16 logits.
            let (shape_bf16, bf16) = load_ref_in(&dir, "logits_mlx");
            assert_eq!(shape_bf16, shape, "{name}: bf16 reference shape");
            eprintln!(
                "{}",
                logit_dev(&got, &bf16, vocab).row(&format!(
                    "Q4 vs mlx-lm bf16 ({name}, {} positions; context, not gated)",
                    ids.len()
                ))
            );
            let d: LogitDev = logit_dev(&got, &want, vocab);
            eprintln!(
                "{}",
                d.row(&format!(
                    "Q4 vs mlx-lm float32 ({name}, {} positions; gated)",
                    ids.len()
                ))
            );
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

/// Attribution for a gap between tessl and mlx-lm on the same 4-bit weights.
/// `make_mlx_q4.py` also dumps mlx-lm run in FLOAT32 (`logits_mlx_f32.npy`: every
/// floating parameter cast to float32, dtype checked at dump time), the exact
/// computation on these weights up to float32 accumulation. This prints, per
/// prompt, through the shared harness:
///
/// * tessl Q4 against mlx-lm float32;
/// * mlx-lm bfloat16 (the file `mlx_q4_2b_matches_mlx_lm_logits` compares with)
///   against mlx-lm float32;
/// * tessl Q4 against mlx-lm bfloat16 (the gated figure, for reference);
///
/// and a verdict line saying which of the first two is closer on each of max
/// deviation, top-1 misses and mean KL. tessl's Q4 forward rounds each GEMM's
/// input to bf16 and keeps the residual stream and every accumulation in f32
/// (`Format::precision`, `Acts`; `Q4Weight::matmul` takes a bf16 `x` and writes
/// f32); mlx-lm bf16 rounds every layer output. If tessl is as close to the
/// float32 reference as mlx-lm bf16 is, or closer, the misses against mlx-lm bf16
/// are mlx-lm's rounding, not tessl's. A report, not a gate: it asserts only that
/// both reference files exist, are finite and have the right shape. The bound of
/// the gated test is the owner's to change with this evidence.
#[test]
#[ignore]
fn mlx_q4_2b_attribution_against_mlx_lm_float32() {
    with_gpu(|rt| {
        let (model, _) = load_real(rt);
        let vocab = model.config().vocab as usize;
        for name in ["default", "text"] {
            let dir = q4_dir().join("ref").join(name);
            let ids = ref_ids(&dir);
            let (shape, bf16) = load_ref_in(&dir, "logits_mlx");
            let (shape32, f32_ref) = load_ref_in(&dir, "logits_mlx_f32");
            assert_eq!(shape, vec![ids.len(), vocab], "{name}: bf16 reference shape");
            assert_eq!(shape32, shape, "{name}: float32 reference shape");
            let got = model.forward(&ids, false).unwrap().logits;
            let mlx_bf16: Vec<f32> = bf16.iter().map(|&x| x as f32).collect();
            let tessl_vs_f32 = logit_dev(&got, &f32_ref, vocab);
            let bf16_vs_f32 = logit_dev(&mlx_bf16, &f32_ref, vocab);
            let tessl_vs_bf16 = logit_dev(&got, &bf16, vocab);
            for (label, d) in [
                ("tessl Q4 vs mlx-lm float32", &tessl_vs_f32),
                ("mlx-lm bf16 vs mlx-lm float32", &bf16_vs_f32),
                ("tessl Q4 vs mlx-lm bf16 (gated)", &tessl_vs_bf16),
            ] {
                assert!(d.is_finite(), "{name}: {label}: non-finite");
                eprintln!("{}", d.row(&format!("[{name} T={}] {label}", ids.len())));
            }
            let cmp = |what: &str, tessl: f64, mlx: f64| {
                let v = if tessl < mlx {
                    "closer"
                } else if tessl == mlx {
                    "equal"
                } else {
                    "farther"
                };
                format!("{what}: tessl {tessl:.3e} vs mlx bf16 {mlx:.3e} ({v})")
            };
            eprintln!(
                "VERDICT [{name}] tessl Q4 against the float32 reference, relative to mlx-lm bf16 against it:\n  {}\n  {}\n  {}",
                cmp("max logit deviation (rel)", tessl_vs_f32.max_rel(), bf16_vs_f32.max_rel()),
                cmp(
                    "top-1 misses",
                    tessl_vs_f32.top1_misses() as f64,
                    bf16_vs_f32.top1_misses() as f64
                ),
                cmp("mean KL", tessl_vs_f32.mean_kl, bf16_vs_f32.mean_kl),
            );
        }
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

// ------------------------------------------------- adversarial lane (w23) ---
//
// Device tests added by the adversarial review of 485dbf5; the lead runs them
// under the machine lock. Tests marked `#[ignore = "BREAK: ..."]` assert what a
// fail-closed loader does and fail today; run them with `--ignored`.

/// Make every quantized matrix of `net` exactly representable in bf16 once
/// dequantized (power-of-two scales, bias a whole multiple of the scale, so
/// `w = scale * (q - k)` is a small integer times a power of two), and every
/// norm a multiple of 2^-6 (so the converter's bf16 `1 + w` is exact). The
/// dequantized dense checkpoint is then bf16 bit for bit, and the Q4 and dense
/// bf16 models differ only in accumulation order. The GDN `in_proj_b` and
/// `in_proj_a` windows get large, opposite-signed weights so a window placed in
/// the wrong rows moves the logits by far more than the accumulation noise.
fn make_exact(net: &mut Net, seed: u64) {
    let mut rng = Rng(seed);
    let mut fix = |name: &str, m: &mut QMat| {
        let (scale_exp, k): (&[i32], &[f32]) = if name.contains("in_proj_b") {
            (&[-5], &[2.0])
        } else if name.contains("in_proj_a") {
            (&[-5], &[12.0])
        } else {
            (&[-8, -7, -6], &[0.0, 2.0, 8.0])
        };
        for (s, b) in m.scales.iter_mut().zip(m.biases.iter_mut()) {
            let sc = 2.0f32.powi(scale_exp[rng.next() as usize % scale_exp.len()]);
            *s = sc;
            *b = -k[rng.next() as usize % k.len()] * sc;
        }
    };
    for (name, m) in net.mats.iter_mut() {
        fix(name, m);
    }
    fix("lm_head", &mut net.head);
    for (_, w) in net.norms.iter_mut() {
        for v in w.iter_mut() {
            *v = ((rng.next() % 39) as f32 - 19.0) / 64.0;
        }
    }
}

/// Swap two same-shaped tensors' payloads under their names.
fn swap_payloads(entries: &mut [Entry], a: &str, b: &str) {
    let (ia, ib) = (
        entries.iter().position(|e| e.0 == a).unwrap(),
        entries.iter().position(|e| e.0 == b).unwrap(),
    );
    let (da, ds, db) = (entries[ia].1, entries[ia].2.clone(), entries[ia].3.clone());
    entries[ia].1 = entries[ib].1;
    entries[ia].2 = entries[ib].2.clone();
    entries[ia].3 = entries[ib].3.clone();
    entries[ib].1 = da;
    entries[ib].2 = ds;
    entries[ib].3 = db;
}

/// Attack 3 (packed-window concatenation): with weights exact in bf16, the Q4
/// model and the dense Bf16 model on the same values must agree to
/// accumulation-order noise, at every row count the routing distinguishes.
/// Control: the same Q4 checkpoint with `in_proj_a` and `in_proj_b` swapped in
/// every GDN layer must miss by at least four times the matched error, so the
/// bound has the power to see a window in the wrong rows.
#[test]
fn adv_mlx_q4_exact_weights_match_dense_bf16_and_a_swapped_window_does_not() {
    const MATCH_BOUND: f64 = 1e-2;
    with_gpu(|rt| {
        let mut net = Net::new(21);
        make_exact(&mut net, 22);
        let q4 = load_q4(rt, &net.mlx_entries()).unwrap();
        // BF16 matrices: the bf16 dense loader reads BF16 only. `make_exact`
        // makes the dequantized values exact in bf16, so nothing is rounded.
        let dense_st = safetensors(&net.dense_entries_bf16());
        let dense = Qwen35Model::load(rt, &dense_st, "model.", config(), Precision::Bf16).unwrap();

        let mut swapped = net.mlx_entries();
        for l in [0usize, 2] {
            for sfx in ["weight", "scales", "biases"] {
                swap_payloads(
                    &mut swapped,
                    &format!("language_model.model.layers.{l}.linear_attn.in_proj_a.{sfx}"),
                    &format!("language_model.model.layers.{l}.linear_attn.in_proj_b.{sfx}"),
                );
            }
        }
        let bad = load_q4(rt, &swapped).unwrap();

        let (mut worst, mut worst_ctl) = (0.0f64, f64::MAX);
        for t in [1usize, 2, 7, 8, 9, 33, 70] {
            let ids = seq(t);
            let want = dense.forward(&ids, false).unwrap().logits;
            let got = q4.forward(&ids, false).unwrap().logits;
            let ctl = bad.forward(&ids, false).unwrap().logits;
            assert!(got.iter().all(|x| x.is_finite()), "t={t}: non-finite logits");
            let (r, c) = (rel(&got, &want), rel(&ctl, &want));
            eprintln!("t={t}: exact match err {r:.3e}, swapped-window control {c:.3e}");
            assert!(r <= MATCH_BOUND, "t={t}: Q4 vs dense bf16 on exact weights: {r:.3e}");
            worst = worst.max(r);
            worst_ctl = worst_ctl.min(c);
        }
        assert!(
            worst_ctl >= 4.0 * worst.max(1e-3),
            "the control cannot tell a swapped window from noise: control {worst_ctl:.3e} vs match {worst:.3e}"
        );
    });
}

/// Attack 5 (routing): every row count around both cuts, and large ragged
/// ones, run Q4 and agree with the dequantized F32 model. 7 / 8 are the last
/// small-M sizes, 9 the first TensorOps one; 33, 129 and 300 are ragged against
/// the 32-row tile and past the GDN chunk.
#[test]
fn adv_mlx_q4_routing_boundaries_and_ragged_row_counts_match_dense() {
    with_gpu(|rt| {
        let net = Net::new(31);
        let q4 = load_q4(rt, &net.mlx_entries()).unwrap();
        let dense_st = safetensors(&net.dense_entries());
        let dense = Qwen35Model::load(rt, &dense_st, "model.", config(), Precision::F32).unwrap();
        for t in [1usize, 2, 7, 8, 9, 10, 31, 32, 33, 64, 65, 129, 300] {
            let ids = seq(t);
            let got = q4.forward(&ids, false).unwrap().logits;
            let want = dense.forward(&ids, false).unwrap().logits;
            let r = rel(&got, &want);
            assert!(r <= TINY_BOUND, "t={t}: rel err {r:.3e} > {TINY_BOUND:.0e}");
        }
    });
}

/// Attack 1 (NaN / inf scales, biases and norms): a non-finite scale or bias
/// would be copied to the bank unchecked and a non-finite norm to the kernel,
/// and the model would answer with NaN logits. A loader that fails closed
/// refuses each by tensor name. (The dense F32/bf16 loaders do not check
/// either; this is the Q4 loader's own boundary.)
#[test]
fn adv_mlx_q4_nonfinite_scales_biases_and_norms_are_refused() {
    with_gpu(|rt| {
        let net = Net::new(41);
        let ok = net.mlx_entries();
        let poison = |name: &str, v: f32| {
            let mut e = ok.clone_entries();
            let hit = e.iter_mut().find(|e| e.0 == name).unwrap();
            hit.3[..2].copy_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
            e
        };
        for (name, v) in [
            ("language_model.model.layers.0.mlp.gate_proj.scales", f32::NAN),
            ("language_model.model.layers.0.mlp.gate_proj.scales", f32::INFINITY),
            (
                "language_model.model.layers.1.self_attn.o_proj.biases",
                f32::NEG_INFINITY,
            ),
            ("language_model.model.embed_tokens.scales", f32::NAN),
            ("language_model.lm_head.biases", f32::INFINITY),
            ("language_model.model.norm.weight", f32::NAN),
            ("language_model.model.layers.0.input_layernorm.weight", f32::INFINITY),
        ] {
            let r = load_q4(rt, &poison(name, v));
            let e = r.err().unwrap_or_else(|| panic!("{name} = {v} loaded"));
            assert!(
                e.contains(name.rsplit('.').next().unwrap()) && e.contains("finite"),
                "{name}: {e}"
            );
        }
    });
}

/// Attack 1 (a huge declared shape on a tiny file; the config is as untrusted
/// as the file): a config whose vocabulary is vast must be refused against the
/// header before the first bank is allocated. `Loader::mlx_q4_named` sizes and
/// allocates the whole bank (`Q4Weight::alloc`) and only then runs
/// `check_part` on the header, so a tiny file under a hostile config first
/// asks the device for up to its max buffer length.
#[test]
fn adv_mlx_q4_a_huge_config_over_a_tiny_file_fails_before_allocating() {
    with_gpu(|rt| {
        let net = Net::new(51);
        let st = safetensors(&net.mlx_entries());
        let cfg = Qwen35Config {
            vocab: 1 << 22,
            ..config()
        };
        rt.reset_peak_allocated_bytes();
        let before = rt.peak_allocated_bytes();
        let e = Qwen35Model::load_mlx_q4(rt, &st, cfg, QUANT_JSON)
            .err()
            .expect("loaded");
        assert!(e.contains("embed_tokens"), "{e}");
        let grew = rt.peak_allocated_bytes().saturating_sub(before);
        // The embedding at 2^22 x 64 is 128 MiB packed; the whole tiny file is
        // a few hundred KiB.
        assert!(
            grew < 8 << 20,
            "the refused load still allocated {grew} bytes on the device"
        );
    });
}

/// The dense loaders have the Q4 loader's order problem, and the fix is in the
/// shared `Loader`: a config that sizes a table or a packed projection is
/// checked against the file's header before the tensor is allocated. A config
/// whose vocabulary (the tables) or intermediate width (the MLP projections) is
/// vast over a tiny dense file is refused with the tensor's name and the device
/// allocates next to nothing, at both dense precisions. Each precision reads a
/// checkpoint stored as it requires (BF16 matrices for `Precision::Bf16`, which
/// loads BF16 only; F32 for `Precision::F32`), so the only thing wrong with
/// the file is the config. The fail-before run is the branch
/// `gp/w23-q4-failbefore` (this test on the commit before the fix).
#[test]
fn dense_load_with_a_huge_config_over_a_tiny_file_fails_before_allocating() {
    with_gpu(|rt| {
        let net = Net::new(52);
        let (st_bf16, st_f32) = (
            safetensors(&net.dense_entries_bf16()),
            safetensors(&net.dense_entries()),
        );
        for precision in [Precision::Bf16, Precision::F32] {
            let st = if precision == Precision::Bf16 {
                &st_bf16
            } else {
                &st_f32
            };
            for (cfg, needle) in [
                (
                    Qwen35Config {
                        vocab: 1 << 22,
                        ..config()
                    },
                    "embed_tokens",
                ),
                (
                    Qwen35Config {
                        intermediate: 1 << 22,
                        ..config()
                    },
                    "gate_proj",
                ),
            ] {
                rt.reset_peak_allocated_bytes();
                let before = rt.peak_allocated_bytes();
                let e = Qwen35Model::load(rt, st, "model.", cfg, precision)
                    .err()
                    .expect("loaded");
                assert!(e.contains(needle), "{precision:?}: {e}");
                let grew = rt.peak_allocated_bytes().saturating_sub(before);
                assert!(
                    grew < 8 << 20,
                    "{precision:?} {needle}: the refused load still allocated {grew} bytes on the device"
                );
            }
        }
    });
}

/// What the bf16 dense loader accepts did not change: an F32 matrix in a
/// `Precision::Bf16` load is refused with the dtype message (it is not cast),
/// and a valid BF16 checkpoint loads. The header-first shape check
/// (`Loader::expect_shape`) checks shape only, so it neither adds nor removes
/// a dtype refusal: the same file with a hostile config is refused for its
/// shape, and with the right config for its dtype.
#[test]
fn dense_bf16_load_refuses_an_f32_matrix_as_it_always_did() {
    with_gpu(|rt| {
        let net = Net::new(53);
        let f32_file = safetensors(&net.dense_entries());
        let e = Qwen35Model::load(rt, &f32_file, "model.", config(), Precision::Bf16)
            .err()
            .expect("an F32 matrix loaded at Bf16");
        assert!(
            e.contains("embed_tokens.weight") && e.contains("expected BF16, found F32"),
            "{e}"
        );
        let bf16_file = safetensors(&net.dense_entries_bf16());
        Qwen35Model::load(rt, &bf16_file, "model.", config(), Precision::Bf16).unwrap();
        // And F32 still reads a BF16 file by widening (the f32 reads take either).
        Qwen35Model::load(rt, &bf16_file, "model.", config(), Precision::F32).unwrap();
    });
}

/// Attack 2 (a conv layout of one convention, norms of the other): the loader
/// reads the convention from `conv1d.weight`'s shape alone. A checkpoint with
/// the sanitized `[c, k, 1]` conv but norms that were never shifted (a
/// third-party conversion, or mlx-vlm's `should_offset_norm_weight` skipping
/// them) loads without complaint and is a different model: this test pins that
/// it is silent and far off, so a mitigation (a norm-statistics plausibility
/// check, or refusing the HF key spelling under the MLX conv layout) has a
/// failing test to turn.
#[test]
fn adv_mlx_q4_conv_and_norm_convention_disagreement_loads_silently_as_a_wrong_model() {
    with_gpu(|rt| {
        let net = Net::new(61);
        let dense_st = safetensors(&net.dense_entries());
        let dense = Qwen35Model::load(rt, &dense_st, "model.", config(), Precision::F32).unwrap();
        // Norms left as `w` (no +1) beside the `[c, k, 1]` conv layout.
        let mut e = net.mlx_entries();
        for (n, v) in &net.norms {
            let hit = e
                .iter_mut()
                .find(|x| x.0 == format!("language_model.model.{n}"))
                .unwrap();
            hit.3 = bf16_bytes(v);
        }
        let wrong = load_q4(rt, &e).expect("a convention disagreement is undetectable from the header");
        let ids = seq(9);
        let got = wrong.forward(&ids, false).unwrap().logits;
        let want = dense.forward(&ids, false).unwrap().logits;
        let r = rel(&got, &want);
        eprintln!("convention disagreement: rel err {r:.3e}");
        assert!(r > 0.3, "expected garbage, got rel err {r:.3e}");
    });
}

/// Attack 7 (footprint report): the host peak is the process's LIFETIME max
/// (`ri_lifetime_max_phys_footprint`), so anything allocated earlier in the
/// process hides the load's own peak: the report's "after" cannot be below
/// "before", and a load that stays under an earlier peak shows no growth
/// whatever it used. Characterizes the property the doc comment states.
#[test]
fn adv_mlx_q4_footprint_report_is_a_lifetime_max_an_earlier_allocation_hides_the_load() {
    with_gpu(|rt| {
        // Touch 384 MiB on the host, drop it: the lifetime max keeps it.
        let ballast = vec![1u8; 384 << 20];
        assert!(ballast.iter().step_by(4096).all(|&b| b == 1));
        drop(ballast);
        let net = Net::new(71);
        let st = safetensors(&net.mlx_entries());
        let (_, r) = Qwen35Model::load_mlx_q4(rt, &st, config(), QUANT_JSON).unwrap();
        const MIB: u64 = 1 << 20;
        // What a lifetime maximum guarantees, each part of it:
        // 1. the earlier 384 MiB is still in the figure read before the load;
        assert!(
            r.peak_footprint_before >= 384 * MIB,
            "before = {}",
            r.peak_footprint_before
        );
        // 2. it never goes down across the load;
        assert!(r.peak_footprint_after >= r.peak_footprint_before, "{r:?}");
        // 3. a load of a few hundred KiB adds almost nothing to it: the report
        //    after is the earlier peak plus at most the load's own small excess
        //    (the host's incidental allocations in between, not 384 MiB), so the
        //    figure says nothing about what the load itself used;
        let grew = r.peak_footprint_after - r.peak_footprint_before;
        assert!(
            grew <= 64 * MIB,
            "a tiny load raised the lifetime peak by {grew} B; before {} after {}",
            r.peak_footprint_before,
            r.peak_footprint_after
        );
        // 4. and a second load in the same process reads at least what the
        //    first left behind (the counter is the process's, not the load's).
        let (_, r2) = Qwen35Model::load_mlx_q4(rt, &st, config(), QUANT_JSON).unwrap();
        assert!(
            r2.peak_footprint_before >= r.peak_footprint_after,
            "second load's before {} < first load's after {}",
            r2.peak_footprint_before,
            r.peak_footprint_after
        );
    });
}
