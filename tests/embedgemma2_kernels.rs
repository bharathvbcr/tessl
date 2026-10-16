//! EmbeddingGemma 2's kernels against transformers, then against an f64
//! reference at the real shapes.
//!
//! 1. **The references are transformers.** `tests/fixtures/embedgemma2/` holds
//!    goldens from `scripts/gen_embedgemma2_fixtures.py` (transformers 5.19's
//!    own masks, `eager_attention_forward`, norms, RoPE and PLE modules). The
//!    f64 references below are held to those first, on the CPU.
//! 2. **The kernels are held to the references** at the fixture shapes and at
//!    the model's real ones (window 512, head dims 256/512, sequences longer
//!    than the window, ragged batches), and to the fixtures directly.
//! 3. **Adversarial**: a length-1 row, a row whose length is 0 or past the
//!    padded length, a window wider than the sequence, a batch row's output
//!    against the same sequence run alone, bitwise repeatability, and every
//!    host-side refusal.
//!
//! Bounds, fixed before any run: an f32 kernel against an f64 reference within
//! `2e-5 * max|ref|` (one f32 rounding per step over at most ~8k-term sums);
//! transformers' f32 output against the f64 reference within `1e-5 * max|ref|`.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::embedgemma2::{attn_ref, gelu_tanh, l2_normalize, lin, pool_reference, rms, rope, Attn};
use common::{tensor_f32, with_gpu, SplitMix};
use tessl::embedgemma2::{
    encoder_attn, encoder_attn_with, l2_normalize_rows, segment_mean_rows, upload_segments, EmbedGemma2Config,
    EncoderAttnDims, EncoderAttnKernel,
};
use tessl::gemm::{gemm, GemmBackend};
use tessl::nn::{self, mlp_gelu_tanh, rms_norm_f32, rms_norm_residual_add_f32, scale_f32_inplace};
use tessl::npy::read_npy;
use tessl::qwen35::{self, AttnShape, AttnTargets, Cols, QkvColumns};
use tessl::tensor::{DType, GpuBuffer};
use tessl::GpuRuntime;

const KERNEL_REL: f64 = 2e-5;
const FIXTURE_REL: f64 = 1e-5;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2")
}

fn load(name: &str) -> (Vec<usize>, Vec<f64>) {
    let path = fixture_dir().join(format!("eg2_{name}.npy"));
    let a = read_npy(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
    let data = if let Ok(s) = a.f32_slice() {
        s.iter().map(|&v| f64::from(v)).collect()
    } else if let Some(i) = a.data_i64.as_ref() {
        i.iter().map(|&v| v as f64).collect()
    } else {
        a.f64_slice().expect("f32, i64 or f64 fixture").to_vec()
    };
    (a.shape.clone(), data)
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn buf_f32(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).unwrap();
    b.write_f32(data);
    b
}

/// `x` rounded to bf16, round to nearest even, widened back (finite inputs).
fn bf16_rne(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// `data` as a bf16 device buffer, each element rounded with [`bf16_rne`].
fn bf16_buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 2).unwrap();
    let mut bits: Vec<u16> = data.iter().map(|&x| (bf16_rne(x).to_bits() >> 16) as u16).collect();
    bits.resize(data.len().max(1), 0);
    b.write_bf16_bits(&bits);
    b
}

fn buf_u32(rt: &Arc<GpuRuntime>, data: &[u32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).unwrap();
    b.write_u32(data);
    b
}

/// `max |got - want|` over the listed indices, against `rel * max |want|`.
fn assert_close(what: &str, got: &[f64], want: &[f64], idx: &[usize], rel: f64) {
    let scale = idx.iter().map(|&i| want[i].abs()).fold(0.0, f64::max).max(1e-30);
    let (mut worst, mut at) = (0.0f64, 0usize);
    for &i in idx {
        let e = (got[i] - want[i]).abs();
        if !e.is_finite() || e > worst {
            worst = if e.is_finite() { e } else { f64::INFINITY };
            at = i;
        }
    }
    eprintln!(
        "{what}: max abs error {worst:.3e} = {:.2e} of max |ref| {scale:.3e}",
        worst / scale
    );
    assert!(
        worst <= rel * scale,
        "{what}: max abs error {worst:.3e} at {at} exceeds {rel:.0e} * {scale:.3e} (got {}, want {})",
        got[at],
        want[at]
    );
}

// ---------------------------------------------------------------------------
// 1. The references are transformers
// ---------------------------------------------------------------------------

fn attn_fixture(name: &str) -> (Attn, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (qs, q) = load(&format!("{name}_q"));
    let (ks, k) = load(&format!("{name}_k"));
    let (_, v) = load(&format!("{name}_v"));
    let (_, out) = load(&format!("{name}_out"));
    let (_, lens) = load(&format!("{name}_lens"));
    let (_, window) = load(&format!("{name}_window"));
    let a = Attn {
        b: qs[0],
        t: qs[1],
        h: qs[2],
        hkv: ks[2],
        d: qs[3],
        window: if window[0] < 0.0 { 0 } else { window[0] as usize },
        lens: lens.iter().map(|&l| l as usize).collect(),
    };
    // transformers returns [B, T, H, D] after its transpose back.
    (a, q, k, v, out)
}

fn valid_rows(a: &Attn) -> Vec<usize> {
    let mut idx = Vec::new();
    for b in 0..a.b {
        for t in 0..a.lens[b].min(a.t) {
            let base = (b * a.t + t) * a.h * a.d;
            idx.extend(base..base + a.h * a.d);
        }
    }
    idx
}

#[test]
fn attention_reference_matches_transformers() {
    for name in ["swa", "global", "swa_tiny"] {
        let (a, q, k, v, want) = attn_fixture(name);
        let got = attn_ref(&a, &q, &k, &v, None);
        assert_close(name, &got, &want, &valid_rows(&a), FIXTURE_REL);
    }
}

#[test]
fn qkv_norm_rope_reference_matches_transformers() {
    // Near position 0 only: far from it transformers' f32 angles are the
    // larger error, which `qkv_columns_norm_rope_far_positions` bounds instead.
    for name in ["qkv256", "qkv512"] {
        let (qs, q) = load(&format!("{name}_q_in"));
        let (ks, k) = load(&format!("{name}_k_in"));
        let (_, v) = load(&format!("{name}_v_in"));
        let (_, qw) = load(&format!("{name}_q_norm_w"));
        let (_, kw) = load(&format!("{name}_k_norm_w"));
        let (_, pos0) = load(&format!("{name}_meta"));
        let theta = if qs[3] == 256 { 1e4 } else { 1e6 };
        let (t, d) = (qs[1], qs[3]);
        let per = |x: &[f64], heads: usize, w: Option<&[f64]>, rotate: bool| -> Vec<f64> {
            let mut out = Vec::with_capacity(x.len());
            for ti in 0..t {
                for hh in 0..heads {
                    let o = (ti * heads + hh) * d;
                    let n = rms(&x[o..o + d], w);
                    out.extend(if rotate {
                        rope(&n, pos0[0] + ti as f64, theta)
                    } else {
                        n
                    });
                }
            }
            out
        };
        let checks = [
            ("q", per(&q, qs[2], Some(&qw), true), load(&format!("{name}_q_out")).1),
            ("k", per(&k, ks[2], Some(&kw), true), load(&format!("{name}_k_out")).1),
            ("v", per(&v, ks[2], None, false), load(&format!("{name}_v_out")).1),
        ];
        for (what, got, want) in checks {
            let idx: Vec<usize> = (0..want.len()).collect();
            assert_close(&format!("{name} {what}"), &got, &want, &idx, 5e-6);
        }
    }
}

/// The fixture PLE at width 64: projection, scale, norm, then the block.
fn ple_reference() -> (Vec<f64>, Vec<f64>) {
    let (es, embeds) = load("ple_embeds");
    let (_, hidden) = load("ple_hidden");
    let (pws, proj_w) = load("ple_proj_w");
    let (_, proj_norm_w) = load("ple_proj_norm_w");
    let (_, gate_w) = load("ple_gate_w");
    let (_, out_w) = load("ple_out_proj_w");
    let (_, post_w) = load("ple_post_norm_w");
    let (_, layer) = load("ple_layer");
    let (rows, h) = (es[0] * es[1], es[2]);
    let n_layers = pws[0] / h;
    let li = layer[0] as usize;
    let mut per_layer = Vec::with_capacity(rows * n_layers * h);
    let mut block = Vec::with_capacity(rows * h);
    for r in 0..rows {
        let e = &embeds[r * h..(r + 1) * h];
        let p: Vec<f64> = lin(&proj_w, e, n_layers * h, h)
            .iter()
            .map(|v| v * (h as f64).powf(-0.5))
            .collect();
        let mut mine = Vec::new();
        for l in 0..n_layers {
            let n = rms(&p[l * h..(l + 1) * h], Some(&proj_norm_w));
            if l == li {
                mine = n.clone();
            }
            per_layer.extend(n);
        }
        let x = &hidden[r * h..(r + 1) * h];
        let g: Vec<f64> = lin(&gate_w, x, h, h)
            .iter()
            .zip(&mine)
            .map(|(g, m)| gelu_tanh(*g) * m)
            .collect();
        let y = rms(&lin(&out_w, &g, h, h), Some(&post_w));
        block.extend(x.iter().zip(&y).map(|(a, b)| a + b));
    }
    (per_layer, block)
}

#[test]
fn ple_reference_matches_transformers() {
    let (per_layer, block) = ple_reference();
    let want_pl = load("ple_per_layer").1;
    let want_block = load("ple_block_out").1;
    assert_close(
        "per_layer",
        &per_layer,
        &want_pl,
        &(0..want_pl.len()).collect::<Vec<_>>(),
        5e-6,
    );
    assert_close(
        "block",
        &block,
        &want_block,
        &(0..want_block.len()).collect::<Vec<_>>(),
        5e-6,
    );
}

#[test]
fn pool_reference_matches_transformers() {
    let (xs, x) = load("pool_x");
    let (_, lens) = load("pool_lens");
    let lens: Vec<usize> = lens.iter().map(|&l| l as usize).collect();
    let got = pool_reference(&x, &lens, xs[1], xs[2]);
    let want = load("pool_out").1;
    assert_close("pool", &got, &want, &(0..want.len()).collect::<Vec<_>>(), 5e-6);
}

// ---------------------------------------------------------------------------
// 2. The kernels
// ---------------------------------------------------------------------------

/// Both kernels hold the same contract; every attention test runs on each.
const KERNELS: [EncoderAttnKernel; 2] = [EncoderAttnKernel::Rows, EncoderAttnKernel::Tiled];

/// The kernels whose outputs are compared with each other bitwise (batch
/// independence, repeatability), which holds for the bf16-operand kernel too:
/// it has no f32 reference to share a bound with, but its arithmetic does not
/// depend on the batch or the run.
const BITWISE_KERNELS: [EncoderAttnKernel; 3] = [
    EncoderAttnKernel::Rows,
    EncoderAttnKernel::Tiled,
    EncoderAttnKernel::TiledBf16,
];

/// Whether `kernel` is compiled for head dim `d` (the bf16 kernel is not for 32).
fn compiled(kernel: EncoderAttnKernel, d: usize) -> bool {
    kernel != EncoderAttnKernel::TiledBf16 || d != 32
}

/// The head dims the structural tests (batch-independence, masking, length
/// clamping) cover: the text encoder's sliding layers (256), the vision
/// tower's (64) and BERT-small's (32). D=512 has its own cases.
const HEAD_DIMS: [usize; 3] = [256, 64, 32];

fn run_attn(
    rt: &Arc<GpuRuntime>,
    kernel: EncoderAttnKernel,
    a: &Attn,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    lens: &[u32],
) -> Vec<f64> {
    // `TiledBf16` reads bf16 operands: the f32 inputs are rounded here, on the
    // host, with `bf16_rne`.
    let (qb, kb, vb) = if kernel == EncoderAttnKernel::TiledBf16 {
        (bf16_buf(rt, q), bf16_buf(rt, k), bf16_buf(rt, v))
    } else {
        (buf_f32(rt, q), buf_f32(rt, k), buf_f32(rt, v))
    };
    let o = rt.alloc_buffer(q.len().max(1) * 4).unwrap();
    // Poison the output so an unwritten element cannot pass as zero.
    o.write_f32(&vec![f32::NAN; q.len()]);
    let lb = buf_u32(rt, lens);
    encoder_attn_with(
        rt,
        &qb,
        &kb,
        &vb,
        &o,
        &lb,
        EncoderAttnDims {
            batch: a.b as u32,
            seq: a.t as u32,
            heads: a.h as u32,
            heads_kv: a.hkv as u32,
            head_dim: a.d as u32,
            window: a.window as u32,
            scale: 1.0,
        },
        kernel,
    )
    .unwrap_or_else(|e| panic!("{kernel:?}: {e}"));
    rt.synchronize().unwrap();
    o.read_f32()[..q.len()].iter().map(|&x| f64::from(x)).collect()
}

fn padding_is_zero(what: &str, a: &Attn, got: &[f64]) {
    for b in 0..a.b {
        for t in a.lens[b].min(a.t)..a.t {
            let base = (b * a.t + t) * a.h * a.d;
            for (i, &g) in got[base..base + a.h * a.d].iter().enumerate() {
                assert!(
                    g == 0.0 && g.is_sign_positive(),
                    "{what}: padding row b={b} t={t} elem {i} is {g}, not +0"
                );
            }
        }
    }
}

#[test]
fn encoder_attn_matches_the_fixtures() {
    with_gpu(|rt| {
        for kernel in KERNELS {
            for name in ["swa", "global", "swa_tiny"] {
                let (a, q, k, v, want) = attn_fixture(name);
                let lens: Vec<u32> = a.lens.iter().map(|&l| l as u32).collect();
                let got = run_attn(rt, kernel, &a, &f32s(&q), &f32s(&k), &f32s(&v), &lens);
                let what = format!("{kernel:?} {name}");
                assert_close(&what, &got, &want, &valid_rows(&a), KERNEL_REL);
                padding_is_zero(&what, &a, &got);
            }
        }
    });
}

fn random(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut r = SplitMix::new(seed);
    (0..n).map(|_| (f64::from(r.unit()) * 2.0 - 1.0) * scale).collect()
}

/// The rows a long reference is checked at: the window's edges and both ends.
fn probe_rows(t: usize, window: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = vec![0, 1, 2, t / 2, t - 2, t - 1];
    if window > 0 {
        for c in [
            window - 1,
            window,
            window + 1,
            2 * window,
            2 * window + 1,
            t.saturating_sub(window + 1),
        ] {
            rows.push(c);
        }
    }
    rows.retain(|&r| r < t);
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn check_random(rt: &Arc<GpuRuntime>, kernel: EncoderAttnKernel, a: &Attn, seed: u64) {
    let q = random(a.b * a.t * a.h * a.d, seed, 0.25);
    let k = random(a.b * a.t * a.hkv * a.d, seed + 1, 0.25);
    let v = random(a.b * a.t * a.hkv * a.d, seed + 2, 1.0);
    let lens: Vec<u32> = a.lens.iter().map(|&l| l as u32).collect();
    let got = run_attn(rt, kernel, a, &f32s(&q), &f32s(&k), &f32s(&v), &lens);
    let rows = probe_rows(a.t, a.window);
    let want = attn_ref(a, &q, &k, &v, Some(&rows));
    let mut idx = Vec::new();
    for b in 0..a.b {
        for &t in &rows {
            if t < a.lens[b].min(a.t) {
                let base = (b * a.t + t) * a.h * a.d;
                idx.extend(base..base + a.h * a.d);
            }
        }
    }
    assert_close(
        &format!("{kernel:?} random D={} window={} lens={:?}", a.d, a.window, a.lens),
        &got,
        &want,
        &idx,
        KERNEL_REL,
    );
    padding_is_zero("random", a, &got);
}

#[test]
fn encoder_attn_matches_the_reference_at_the_real_window() {
    with_gpu(|rt| {
        for kernel in KERNELS {
            // Sliding layers: D=256, 4 query heads over 2 KV heads, window 512,
            // one sequence longer than the window on both sides, ragged others.
            check_random(
                rt,
                kernel,
                &Attn {
                    b: 4,
                    t: 1100,
                    h: 4,
                    hkv: 2,
                    d: 256,
                    window: 512,
                    lens: vec![1100, 513, 1, 600],
                },
                11,
            );
            // Full layers: D=512, MQA, every key.
            check_random(
                rt,
                kernel,
                &Attn {
                    b: 2,
                    t: 700,
                    h: 4,
                    hkv: 1,
                    d: 512,
                    window: 0,
                    lens: vec![700, 1],
                },
                12,
            );
            // A window wider than every sequence is the global rule.
            check_random(
                rt,
                kernel,
                &Attn {
                    b: 2,
                    t: 90,
                    h: 4,
                    hkv: 2,
                    d: 256,
                    window: 512,
                    lens: vec![90, 37],
                },
                13,
            );
            // The narrowest window: each query sees itself and one neighbour each side.
            check_random(
                rt,
                kernel,
                &Attn {
                    b: 1,
                    t: 70,
                    h: 2,
                    hkv: 1,
                    d: 256,
                    window: 1,
                    lens: vec![70],
                },
                14,
            );
            // The vision tower's shape: 12 heads of 64, MHA, every key, up to
            // 2520 patches, the shorter image padded behind its live rows.
            check_random(
                rt,
                kernel,
                &Attn {
                    b: 2,
                    t: 2520,
                    h: 12,
                    hkv: 12,
                    d: 64,
                    window: 0,
                    lens: vec![2520, 2007],
                },
                15,
            );
            // Head dim 64 with a window (DistilBERT-like) and ragged lengths.
            check_random(
                rt,
                kernel,
                &Attn {
                    b: 3,
                    t: 700,
                    h: 12,
                    hkv: 12,
                    d: 64,
                    window: 100,
                    lens: vec![700, 650, 1],
                },
                16,
            );
            // BERT-small's head dim.
            check_random(
                rt,
                kernel,
                &Attn {
                    b: 2,
                    t: 600,
                    h: 12,
                    hkv: 12,
                    d: 32,
                    window: 0,
                    lens: vec![600, 333],
                },
                17,
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 2b. The bf16-operand kernel: bounds derived from the rounding
// ---------------------------------------------------------------------------
//
// `EncoderAttnKernel::TiledBf16` rounds Q, K and V to bf16 (the q/k-norm-RoPE
// writer stores them so) and the probability tile P that multiplies V; scores,
// the online softmax, the accumulator and the output are f32. Fixed before the
// first run of these tests. Let u = 2^-8 be bf16's unit roundoff (the bound
// `tests/embedgemma2_tiny.rs` uses); a rounding is a relative error uniform in
// [-u, u], so independent roundings have variance u^2/3 of the value squared.
// For one query row with weights w_j = softmax_j(q . k_j) and output
// o_x = sum_j w_j v_jx, the first-order error of the kernel's output is
//
//   (a) V rounding:         sum_j w_j dv_jx           var (u^2/3) sum_j w_j^2 v_jx^2
//   (b) P rounding: the kernel normalizes by the sum of the rounded weights, so
//       dw_j = w_j (eta_j - sum_k w_k eta_k):
//                           sum_j dw_j v_jx           var (u^2/3) sum_j w_j^2 (v_jx - o_x)^2
//   (c) K rounding: ds_j = sum_i q_i k_ji eps_ji moves the weights by
//       w_j (ds_j - mean_w ds), so the output by sum_j w_j (v_jx - o_x) ds_j:
//                                                     var (u^2/3) sum_j w_j^2 (v_jx - o_x)^2 sum_i q_i^2 k_ji^2
//   (d) Q rounding: the same eps_i for every key, ds_j = sum_i q_i eps_i k_ji:
//       sum_i q_i eps_i Cov_w(v_x, k_i), and |Cov_w| <= sd_w(v_x) sd_w(k_i)
//       (Cauchy-Schwarz), so            var <= (u^2/3) sd_w(v_x)^2 sum_i q_i^2 sd_w(k_i)^2.
//
// Second-order terms are O(ds^2) ~ 1e-4 of the first-order ones here. The
// element-wise standard deviation sigma_x is the square root of the four
// variances' sum (`bf16_model`, in f64 from the exact inputs), and a kernel
// element must lie within 6 sigma_x of the f64 reference plus the f32
// kernels' own `KERNEL_REL * max|ref|` (the accumulation and exp error that
// the bf16 kernel shares): 6 sigma is 2e-9 per element for a Gaussian, and
// the (d) term is an upper bound, so the bound is conservative by design.
//
// A second reference removes (a), (c) and (d): the f64 attention of the
// bf16-ROUNDED inputs. The kernel then differs from it only by (b) and the
// f32 term: within 6 sigma_p (b's alone) plus the same f32 term. That is the
// bound on whether the kernel does the right arithmetic on the bf16 values it
// reads; the first is whether the whole is as accurate as bf16 operands can be.

/// bf16's unit roundoff.
const BF16_U: f64 = 1.0 / 256.0;
/// Standard deviations a kernel element may sit from its reference.
const BF16_SIGMAS: f64 = 6.0;

/// One probed output element: its f64 reference and the standard deviations
/// of the bf16 operand rounding's first-order error (module comment above):
/// all four sources, and P's rounding alone.
struct Bf16Pred {
    idx: usize,
    want: f64,
    sigma: f64,
    sigma_p: f64,
}

fn bf16_model(a: &Attn, q: &[f64], k: &[f64], v: &[f64], rows: Option<&[usize]>) -> Vec<Bf16Pred> {
    let group = a.h / a.hkv;
    let all: Vec<usize> = (0..a.t).collect();
    let rows = rows.unwrap_or(&all);
    let u2 = BF16_U * BF16_U / 3.0;
    let mut out = Vec::new();
    for b in 0..a.b {
        let len = a.lens[b].min(a.t);
        for &i in rows {
            if i >= len {
                continue;
            }
            let lo = if a.window == 0 { 0 } else { i.saturating_sub(a.window) };
            let hi = if a.window == 0 { len } else { len.min(i + a.window + 1) };
            for h in 0..a.h {
                let hk = h / group;
                let qo = ((b * a.t + i) * a.h + h) * a.d;
                let kv = |j: usize| ((b * a.t + j) * a.hkv + hk) * a.d;
                let s: Vec<f64> = (lo..hi)
                    .map(|j| (0..a.d).map(|x| q[qo + x] * k[kv(j) + x]).sum::<f64>())
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut w: Vec<f64> = s.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = w.iter().sum();
                w.iter_mut().for_each(|w| *w /= z);
                // g_j = sum_i q_i^2 k_ji^2.
                let g: Vec<f64> = (lo..hi)
                    .map(|j| (0..a.d).map(|x| (q[qo + x] * k[kv(j) + x]).powi(2)).sum::<f64>())
                    .collect();
                // sum_i q_i^2 sd_w(k_i)^2.
                let mut qk = 0.0;
                for x in 0..a.d {
                    let kbar: f64 = (lo..hi).zip(&w).map(|(j, w)| w * k[kv(j) + x]).sum();
                    let var: f64 = (lo..hi).zip(&w).map(|(j, w)| w * (k[kv(j) + x] - kbar).powi(2)).sum();
                    qk += q[qo + x].powi(2) * var;
                }
                for x in 0..a.d {
                    let o: f64 = (lo..hi).zip(&w).map(|(j, w)| w * v[kv(j) + x]).sum();
                    let (mut ax, mut b1, mut b2, mut vw) = (0.0, 0.0, 0.0, 0.0);
                    for (n, j) in (lo..hi).enumerate() {
                        let vj = v[kv(j) + x];
                        let dv = vj - o;
                        ax += (w[n] * vj).powi(2);
                        let t = (w[n] * dv).powi(2);
                        b1 += t;
                        b2 += t * g[n];
                        vw += w[n] * dv * dv;
                    }
                    out.push(Bf16Pred {
                        idx: qo + x,
                        want: o,
                        sigma: (u2 * (ax + b1 + b2 + vw * qk)).sqrt(),
                        sigma_p: (u2 * b1).sqrt(),
                    });
                }
            }
        }
    }
    out
}

/// The bf16 kernel on `a` (random operands, the f32 kernels' distribution) at
/// `device_lens` (default `a.lens`) against both references, in the bounds
/// above; `rows` limits the reference to those query rows (all when `None`).
/// Returns the largest `|error| / (6 sigma)` seen against each reference.
fn check_bf16(
    rt: &Arc<GpuRuntime>,
    a: &Attn,
    seed: u64,
    rows: Option<&[usize]>,
    device_lens: Option<&[u32]>,
) -> (f64, f64) {
    let f = |v: Vec<f64>| -> Vec<f64> { v.iter().map(|&x| f64::from(x as f32)).collect() };
    let q = f(random(a.b * a.t * a.h * a.d, seed, 0.25));
    let k = f(random(a.b * a.t * a.hkv * a.d, seed + 1, 0.25));
    let v = f(random(a.b * a.t * a.hkv * a.d, seed + 2, 1.0));
    let round = |x: &[f64]| -> Vec<f64> { x.iter().map(|&x| f64::from(bf16_rne(x as f32))).collect() };
    let lens: Vec<u32> = match device_lens {
        Some(l) => l.to_vec(),
        None => a.lens.iter().map(|&l| l as u32).collect(),
    };
    let got = run_attn(
        rt,
        EncoderAttnKernel::TiledBf16,
        a,
        &f32s(&q),
        &f32s(&k),
        &f32s(&v),
        &lens,
    );
    let what = format!("TiledBf16 D={} window={} lens={:?}", a.d, a.window, a.lens);
    assert!(got.iter().all(|x| x.is_finite()), "{what}: non-finite output");
    padding_is_zero(&what, a, &got);
    let exact = bf16_model(a, &q, &k, &v, rows);
    let rounded = bf16_model(a, &round(&q), &round(&k), &round(&v), rows);
    assert_eq!(exact.len(), rounded.len(), "{what}: probes");
    assert!(!exact.is_empty(), "{what}: no live rows probed");
    let scale = exact.iter().map(|p| p.want.abs()).fold(0.0, f64::max);
    let f32_term = KERNEL_REL * scale;
    let (mut worst_exact, mut worst_round) = (0.0f64, 0.0f64);
    for (e, r) in exact.iter().zip(&rounded) {
        let err_e = (got[e.idx] - e.want).abs();
        let err_r = (got[r.idx] - r.want).abs();
        let (bound_e, bound_r) = (BF16_SIGMAS * e.sigma + f32_term, BF16_SIGMAS * r.sigma_p + f32_term);
        assert!(
            err_e <= bound_e,
            "{what} element {}: |{} - {}| = {err_e:.3e} exceeds {BF16_SIGMAS} sigma ({:.3e}) + {f32_term:.1e} against the exact inputs",
            e.idx,
            got[e.idx],
            e.want,
            e.sigma
        );
        assert!(
            err_r <= bound_r,
            "{what} element {}: |{} - {}| = {err_r:.3e} exceeds {BF16_SIGMAS} sigma_p ({:.3e}) + {f32_term:.1e} against the rounded inputs",
            r.idx,
            got[r.idx],
            r.want,
            r.sigma_p
        );
        worst_exact = worst_exact.max(err_e / bound_e);
        worst_round = worst_round.max(err_r / bound_r);
    }
    let max_err = |p: &[Bf16Pred]| p.iter().map(|p| (got[p.idx] - p.want).abs()).fold(0.0, f64::max);
    eprintln!(
        "{what}: max abs error {:.3e} = {:.2e} of max |ref| {scale:.3e} against the exact inputs, {:.3e} against the rounded; \
         worst error / bound {worst_exact:.3} and {worst_round:.3}",
        max_err(&exact),
        max_err(&exact) / scale,
        max_err(&rounded)
    );
    (worst_exact, worst_round)
}

#[test]
fn encoder_attn_bf16_matches_the_reference_within_the_derived_bounds() {
    with_gpu(|rt| {
        // The text encoder's sliding layers: D=256, window 512, a sequence
        // longer than the window on both sides, ragged others.
        let swa = Attn {
            b: 4,
            t: 1100,
            h: 4,
            hkv: 2,
            d: 256,
            window: 512,
            lens: vec![1100, 513, 1, 600],
        };
        check_bf16(rt, &swa, 11, Some(&probe_rows(1100, 512)), None);
        // Its global layers: D=512, MQA, every key.
        let global = Attn {
            b: 2,
            t: 700,
            h: 4,
            hkv: 1,
            d: 512,
            window: 0,
            lens: vec![700, 1],
        };
        check_bf16(rt, &global, 12, Some(&probe_rows(700, 0)), None);
        // The narrowest window.
        let narrow = Attn {
            b: 1,
            t: 70,
            h: 2,
            hkv: 1,
            d: 256,
            window: 1,
            lens: vec![70],
        };
        check_bf16(rt, &narrow, 14, None, None);
        // The vision tower: 12 heads of 64, every key, 2520 patches, the
        // shorter image padded behind its live rows.
        let vision = Attn {
            b: 2,
            t: 2520,
            h: 12,
            hkv: 12,
            d: 64,
            window: 0,
            lens: vec![2520, 2007],
        };
        check_bf16(rt, &vision, 15, Some(&probe_rows(2520, 0)), None);
        // D=64 with a window and ragged lengths.
        let windowed = Attn {
            b: 3,
            t: 700,
            h: 12,
            hkv: 12,
            d: 64,
            window: 100,
            lens: vec![700, 650, 1],
        };
        check_bf16(rt, &windowed, 16, Some(&probe_rows(700, 100)), None);
        // Masked rows beside live ones in one tile, a length-0 row, and
        // lengths past the padded length (device values, clamped to T).
        let ragged = Attn {
            b: 4,
            t: 12,
            h: 2,
            hkv: 1,
            d: 256,
            window: 3,
            lens: vec![5, 0, 12, 1],
        };
        check_bf16(rt, &ragged, 51, None, None);
        let clamped = Attn {
            b: 2,
            t: 40,
            h: 2,
            hkv: 1,
            d: 64,
            window: 7,
            lens: vec![40, 0],
        };
        check_bf16(rt, &clamped, 41, None, Some(&[u32::MAX, 0]));
    });
}

#[test]
fn encoder_attn_row_matches_the_same_sequence_alone_bit_for_bit() {
    with_gpu(|rt| {
        for (kernel, d) in BITWISE_KERNELS
            .into_iter()
            .flat_map(|k| HEAD_DIMS.map(move |d| (k, d)))
            .filter(|&(k, d)| compiled(k, d))
        {
            let (h, hkv, window) = (4usize, 2usize, 512usize);
            let (t_long, len) = (900usize, 333usize);
            let q = f32s(&random(t_long * h * d, 21, 0.25));
            let k = f32s(&random(t_long * hkv * d, 22, 0.25));
            let v = f32s(&random(t_long * hkv * d, 23, 1.0));
            let batched_attn = Attn {
                b: 2,
                t: t_long,
                h,
                hkv,
                d,
                window,
                lens: vec![t_long, len],
            };
            // Row 1 is a copy of row 0's first `len` tokens, then padding garbage.
            let mut qb = q.clone();
            qb.extend_from_slice(&q);
            let mut kb = k.clone();
            kb.extend_from_slice(&k);
            let mut vb = v.clone();
            vb.extend_from_slice(&v);
            let batched = run_attn(rt, kernel, &batched_attn, &qb, &kb, &vb, &[t_long as u32, len as u32]);
            let alone_attn = Attn {
                b: 1,
                t: len,
                h,
                hkv,
                d,
                window,
                lens: vec![len],
            };
            let alone = run_attn(
                rt,
                kernel,
                &alone_attn,
                &q[..len * h * d],
                &k[..len * hkv * d],
                &v[..len * hkv * d],
                &[len as u32],
            );
            let row1 = &batched[t_long * h * d..t_long * h * d + len * h * d];
            for (i, (x, y)) in row1.iter().zip(&alone).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{kernel:?} D={d} element {i}: batched {x} vs alone {y}"
                );
            }
        }
    });
}

#[test]
fn encoder_attn_is_bitwise_repeatable() {
    with_gpu(|rt| {
        for kernel in BITWISE_KERNELS {
            let a = Attn {
                b: 2,
                t: 300,
                h: 4,
                hkv: 1,
                d: 512,
                window: 0,
                lens: vec![300, 120],
            };
            let q = f32s(&random(a.b * a.t * a.h * a.d, 31, 0.25));
            let k = f32s(&random(a.b * a.t * a.hkv * a.d, 32, 0.25));
            let v = f32s(&random(a.b * a.t * a.hkv * a.d, 33, 1.0));
            let first = run_attn(rt, kernel, &a, &q, &k, &v, &[300, 120]);
            let second = run_attn(rt, kernel, &a, &q, &k, &v, &[300, 120]);
            assert!(
                first.iter().zip(&second).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{kernel:?}: two runs differ"
            );
        }
    });
}

#[test]
fn encoder_attn_clamps_device_lengths() {
    // The host cannot see a device length: 0 gives an all-zero row, and one
    // past the padded length behaves as the padded length.
    with_gpu(|rt| {
        for (kernel, d) in KERNELS.into_iter().flat_map(|k| HEAD_DIMS.map(move |d| (k, d))) {
            let a = Attn {
                b: 2,
                t: 40,
                h: 2,
                hkv: 1,
                d,
                window: 7,
                lens: vec![40, 0],
            };
            let q = random(a.b * a.t * a.h * a.d, 41, 0.25);
            let k = random(a.b * a.t * a.hkv * a.d, 42, 0.25);
            let v = random(a.b * a.t * a.hkv * a.d, 43, 1.0);
            let got = run_attn(rt, kernel, &a, &f32s(&q), &f32s(&k), &f32s(&v), &[u32::MAX, 0]);
            let want = attn_ref(&Attn { lens: vec![40, 0], ..a }, &q, &k, &v, None);
            let idx: Vec<usize> = (0..got.len()).collect();
            assert_close(&format!("{kernel:?} D={d} clamped"), &got, &want, &idx, KERNEL_REL);
        }
    });
}

#[test]
fn encoder_attn_fully_masked_rows_are_zeros_beside_live_rows() {
    // A valid query always sees itself, so the only row that runs the key
    // loop with every key masked is a padding query sharing a simdgroup with
    // a live one: at D=256 two rows share a simdgroup (16 lanes each), so an
    // odd length pairs live row len-1 with padding row len, and both iterate
    // the live row's keys. That row's running max never moves off its seed,
    // which is the case a fast-math-folded infinity compare would turn into
    // NaN. Length 0 skips the loop entirely and is checked alongside.
    with_gpu(|rt| {
        for (kernel, d) in KERNELS.into_iter().flat_map(|k| HEAD_DIMS.map(move |d| (k, d))) {
            // The large scale makes scores of order sqrt(D) * scale^2 / 3, so
            // the running max is far from its seed. At D=256 (sd ~85) a row
            // is one-hot and the 2e-5 f32 bound holds with room; at D <= 64
            // two or three keys still compete at |s| ~ 100, where f32
            // rounding of the score alone (2^-24 * |s| * |v|) is a few 1e-5
            // of the output: the rows kernel at D=64, scale 4 measured 6.3e-5
            // against the same f64 reference (conditioning of the problem,
            // not a masking fault). Those dims run the moderate scale 1.
            let big = if d == 256 { 4.0 } else { 1.0 };
            for (scale, window) in [(0.25, 0usize), (big, 0), (0.25, 3), (big, 3)] {
                let a = Attn {
                    b: 4,
                    t: 12,
                    h: 2,
                    hkv: 1,
                    d,
                    window,
                    lens: vec![5, 0, 12, 1],
                };
                let q = random(a.b * a.t * a.h * a.d, 51, scale);
                let k = random(a.b * a.t * a.hkv * a.d, 52, scale);
                let v = random(a.b * a.t * a.hkv * a.d, 53, 1.0);
                let lens: Vec<u32> = a.lens.iter().map(|&l| l as u32).collect();
                let got = run_attn(rt, kernel, &a, &f32s(&q), &f32s(&k), &f32s(&v), &lens);
                let what = format!("{kernel:?} D={d} scale={scale} window={window}");
                assert!(got.iter().all(|x| x.is_finite()), "{what}: non-finite output");
                padding_is_zero(&what, &a, &got);
                let want = attn_ref(&a, &q, &k, &v, None);
                assert_close(&what, &got, &want, &valid_rows(&a), KERNEL_REL);
            }
        }
    });
}

#[test]
fn encoder_attn_refuses_what_it_cannot_run() {
    with_gpu(|rt| {
        let q = rt.alloc_buffer(4 * 8 * 2 * 256 * 4).unwrap();
        let k = rt.alloc_buffer(4 * 8 * 256 * 4).unwrap();
        let v = rt.alloc_buffer(4 * 8 * 256 * 4).unwrap();
        let o = rt.alloc_buffer(4 * 8 * 2 * 256 * 4).unwrap();
        let lens = buf_u32(rt, &[8, 8, 8, 8]);
        let ok = EncoderAttnDims {
            batch: 4,
            seq: 8,
            heads: 2,
            heads_kv: 1,
            head_dim: 256,
            window: 512,
            scale: 1.0,
        };
        encoder_attn(rt, &q, &k, &v, &o, &lens, ok).expect("the baseline call");
        let cases: Vec<(&str, EncoderAttnDims)> = vec![
            ("head dim 128", EncoderAttnDims { head_dim: 128, ..ok }),
            (
                "heads not a multiple",
                EncoderAttnDims {
                    heads: 3,
                    heads_kv: 2,
                    ..ok
                },
            ),
            ("zero kv heads", EncoderAttnDims { heads_kv: 0, ..ok }),
            ("nan scale", EncoderAttnDims { scale: f32::NAN, ..ok }),
            ("seq past the buffers", EncoderAttnDims { seq: 9, ..ok }),
            ("batch past lens", EncoderAttnDims { batch: 5, seq: 6, ..ok }),
        ];
        for (what, dims) in cases {
            assert!(
                encoder_attn(rt, &q, &k, &v, &o, &lens, dims).is_err(),
                "{what}: accepted"
            );
        }
        assert!(
            encoder_attn(rt, &q, &k, &v, &q, &lens, ok).is_err(),
            "o aliasing q: accepted"
        );
        // K/V capacity is derived from their size: fewer than `seq` positions
        // per sequence, or K and V of different capacities, are refused; more
        // positions than `seq` are a wider stride, not an error.
        let kv = |positions: usize| rt.alloc_buffer(4 * positions * 256 * 4).unwrap();
        let (short_k, short_v, long_k, long_v) = (kv(7), kv(7), kv(10), kv(10));
        assert!(
            encoder_attn(rt, &q, &short_k, &short_v, &o, &lens, ok).is_err(),
            "K/V of 7 positions for seq 8: accepted"
        );
        assert!(
            encoder_attn(rt, &q, &k, &long_v, &o, &lens, ok).is_err(),
            "K and V of different capacities: accepted"
        );
        encoder_attn(rt, &q, &long_k, &long_v, &o, &lens, ok).expect("K/V of 10 positions for seq 8");
        // The f32 tiled kernel is compiled for head dims 32, 64, 256 and 512;
        // the bf16 one for 64, 256 and 512 only, and it reads bf16 buffers
        // (two bytes an element), so f32-sized K/V of the same element count
        // are half as many positions to it.
        let small = EncoderAttnDims { head_dim: 64, ..ok };
        let (q64, k64, v64, o64) = (
            rt.alloc_buffer(4 * 8 * 2 * 64 * 4).unwrap(),
            rt.alloc_buffer(4 * 8 * 64 * 4).unwrap(),
            rt.alloc_buffer(4 * 8 * 64 * 4).unwrap(),
            rt.alloc_buffer(4 * 8 * 2 * 64 * 4).unwrap(),
        );
        for kernel in [
            EncoderAttnKernel::Rows,
            EncoderAttnKernel::Tiled,
            EncoderAttnKernel::TiledBf16,
        ] {
            encoder_attn_with(rt, &q64, &k64, &v64, &o64, &lens, small, kernel)
                .unwrap_or_else(|e| panic!("{kernel:?} at head dim 64: {e}"));
        }
        let tiny = EncoderAttnDims { head_dim: 32, ..ok };
        let (q32, k32, v32, o32) = (
            rt.alloc_buffer(4 * 8 * 2 * 32 * 4).unwrap(),
            rt.alloc_buffer(4 * 8 * 32 * 4).unwrap(),
            rt.alloc_buffer(4 * 8 * 32 * 4).unwrap(),
            rt.alloc_buffer(4 * 8 * 2 * 32 * 4).unwrap(),
        );
        for kernel in [EncoderAttnKernel::Rows, EncoderAttnKernel::Tiled] {
            encoder_attn_with(rt, &q32, &k32, &v32, &o32, &lens, tiny, kernel)
                .unwrap_or_else(|e| panic!("{kernel:?} at head dim 32: {e}"));
        }
        assert!(
            encoder_attn_with(rt, &q32, &k32, &v32, &o32, &lens, tiny, EncoderAttnKernel::TiledBf16).is_err(),
            "the bf16 tiled kernel at head dim 32: accepted"
        );
        // bf16 buffers too short for the shape are refused like f32 ones.
        let short_q = rt.alloc_buffer(4 * 8 * 2 * 64 * 2 - 2).unwrap();
        assert!(
            encoder_attn_with(
                rt,
                &short_q,
                &k64,
                &v64,
                &o64,
                &lens,
                small,
                EncoderAttnKernel::TiledBf16
            )
            .is_err(),
            "a bf16 q one element short: accepted"
        );
        let short_lens = buf_u32(rt, &[8]);
        assert!(
            encoder_attn(rt, &q, &k, &v, &o, &short_lens, ok).is_err(),
            "short lens: accepted"
        );
    });
}

/// The norm+RoPE+store outputs of one fixture, widened to f32: q, then k and
/// v from the fixture's first position on.
struct QkvOut {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Run the columns norm+RoPE on fixture `name` at its own positions (the
/// caches hold `pos0 + T` slots, so the kernel writes at the fixture's real
/// positions), storing as `out` (f32, or bf16 widened back exactly).
fn run_qkv(rt: &Arc<GpuRuntime>, name: &str, out: DType) -> QkvOut {
    let elem = if out == DType::BF16 { 2 } else { 4 };
    let (qs, q) = load(&format!("{name}_q_in"));
    let (ks, k) = load(&format!("{name}_k_in"));
    let (_, v) = load(&format!("{name}_v_in"));
    let (t, hq, hkv, d) = (qs[1], qs[2], ks[2], qs[3]);
    let pos0 = load(&format!("{name}_meta")).1[0] as usize;
    let theta = if d == 256 { 1e4f32 } else { 1e6 };
    // One projection row: q | k | v, as the model's packed GEMM writes it.
    let width = (hq + 2 * hkv) * d;
    let mut proj = vec![0f32; t * width];
    for ti in 0..t {
        let row = &mut proj[ti * width..(ti + 1) * width];
        row[..hq * d].copy_from_slice(&f32s(&q[ti * hq * d..(ti + 1) * hq * d]));
        row[hq * d..(hq + hkv) * d].copy_from_slice(&f32s(&k[ti * hkv * d..(ti + 1) * hkv * d]));
        row[(hq + hkv) * d..].copy_from_slice(&f32s(&v[ti * hkv * d..(ti + 1) * hkv * d]));
    }
    let cap = pos0 + t;
    let pb = buf_f32(rt, &proj);
    let qw = buf_f32(rt, &f32s(&load(&format!("{name}_q_norm_w")).1));
    let kw = buf_f32(rt, &f32s(&load(&format!("{name}_k_norm_w")).1));
    let qo = rt.alloc_buffer(t * hq * d * elem).unwrap();
    let ko = rt.alloc_buffer(cap * hkv * d * elem).unwrap();
    let vo = rt.alloc_buffer(cap * hkv * d * elem).unwrap();
    qwen35::attn_qk_norm_rope_columns(
        rt,
        &AttnShape {
            batch: 1,
            seq: t as u32,
            q_heads: hq as u32,
            kv_heads: hkv as u32,
            head_dim: d as u32,
            rotary_dim: d as u32,
        },
        Cols::dense(&pb, width as u32),
        QkvColumns {
            q_head_stride: d as u32,
            k_col: (hq * d) as u32,
            v_col: ((hq + hkv) * d) as u32,
            v_norm: true,
        },
        0.0,
        &qw,
        &kw,
        &AttnTargets {
            q_out: &qo,
            k_cache: &ko,
            v_cache: &vo,
        },
        out,
        pos0 as u32,
        theta,
        1e-6,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    rt.synchronize().unwrap();
    let widen = |b: &GpuBuffer| -> Vec<f32> {
        if out == DType::BF16 {
            b.contents_u16()
                .iter()
                .map(|&h| f32::from_bits(u32::from(h) << 16))
                .collect()
        } else {
            b.read_f32()
        }
    };
    QkvOut {
        q: widen(&qo)[..t * hq * d].to_vec(),
        k: widen(&ko)[pos0 * hkv * d..cap * hkv * d].to_vec(),
        v: widen(&vo)[pos0 * hkv * d..cap * hkv * d].to_vec(),
    }
}

/// [`run_qkv`] at f32, held to the fixture's q/k/v within `rel`.
fn check_qkv(rt: &Arc<GpuRuntime>, name: &str, rel: f64) {
    let got = run_qkv(rt, name, DType::F32);
    let wide = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    for (what, got, want) in [
        ("q", wide(&got.q), load(&format!("{name}_q_out")).1),
        ("k", wide(&got.k), load(&format!("{name}_k_out")).1),
        ("v", wide(&got.v), load(&format!("{name}_v_out")).1),
    ] {
        assert_close(
            &format!("{name} {what}"),
            &got,
            &want,
            &(0..want.len()).collect::<Vec<_>>(),
            rel,
        );
    }
}

/// A K/V buffer's positions per sequence depend on its element size: the same
/// bytes hold twice as many bf16 positions as f32 ones. The writer and the
/// attention both derive the capacity from the buffers, so sizing a bf16 pair
/// as f32 would put batch row `b` at twice the stride the data was written at.
#[test]
fn kv_capacity_follows_the_element_size() {
    with_gpu(|rt| {
        let (batch, hkv, d) = (3u32, 2u32, 64u32);
        let bytes = (batch * hkv * d * 10 * 4) as usize; // ten f32 positions
        let (k, v) = (rt.alloc_buffer(bytes).unwrap(), rt.alloc_buffer(bytes).unwrap());
        assert_eq!(nn::attn_kv_capacity(&k, &v, batch, hkv, d).unwrap(), 10);
        assert_eq!(nn::attn_kv_capacity_bf16(&k, &v, batch, hkv, d).unwrap(), 20);
        let v_short = rt.alloc_buffer(bytes - 2 * (batch * hkv * d) as usize).unwrap();
        assert!(
            nn::attn_kv_capacity_bf16(&k, &v_short, batch, hkv, d).is_err(),
            "K and V of different bf16 capacities: accepted"
        );
    });
}

/// The bf16 targets are the f32 targets rounded once: the same f32 norm, RoPE
/// and V norm run, then the store rounds to nearest even (what a separate
/// cast pass would do), so every bit matches the host's rounding of the f32
/// output. This is the claim that the writer adds no pass and no second
/// rounding.
#[test]
fn qkv_columns_bf16_output_is_the_rounded_f32_output() {
    with_gpu(|rt| {
        for name in ["qkv256", "qkv512"] {
            let f = run_qkv(rt, name, DType::F32);
            let b = run_qkv(rt, name, DType::BF16);
            for (what, got, want) in [("q", &b.q, &f.q), ("k", &b.k, &f.k), ("v", &b.v, &f.v)] {
                assert_eq!(got.len(), want.len(), "{name} {what}");
                for (i, (g, w)) in got.iter().zip(want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        bf16_rne(*w).to_bits(),
                        "{name} {what}[{i}]: bf16 {g} vs rounded f32 {}",
                        bf16_rne(*w)
                    );
                }
            }
        }
    });
}

#[test]
fn qkv_columns_norm_rope_matches_the_fixtures() {
    with_gpu(|rt| {
        check_qkv(rt, "qkv256", KERNEL_REL);
        check_qkv(rt, "qkv512", KERNEL_REL);
    });
}

/// `nn::rope_inv_freq` against the model's own rotary buffers: every
/// entry within one ulp. Where they differ at all it is torch's f32 `pow`
/// that is not correctly rounded (1 of 128 pairs at 256, 3 of 256 at 512).
#[test]
fn rope_inv_freq_matches_the_models_tables() {
    for (d, theta) in [(256u32, 1e4f32), (512, 1e6)] {
        let want = load(&format!("inv_freq_{d}")).1;
        let got = nn::rope_inv_freq(d / 2, d, theta);
        assert_eq!(got.len(), want.len(), "d={d}");
        let mut differ = 0;
        for (p, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let w = w as f32;
            let ulps = (g.to_bits() as i64 - w.to_bits() as i64).abs();
            assert!(ulps <= 1, "d={d} pair {p}: {g:e} vs torch {w:e} ({ulps} ulps)");
            differ += usize::from(ulps != 0);
        }
        assert!(
            differ <= want.len() / 32,
            "d={d}: {differ} of {} pairs differ from torch",
            want.len()
        );
    }
}

/// At position 8000, against transformers (`pos * inv_freq` in f32, then
/// cos/sin). The kernel forms the same f32 angle from the same table, so the
/// bound is the kernel's own f32 error (`KERNEL_REL`) plus the largest angle
/// the table can differ from torch's by: `(pos0 + T) * max_p |ours - torch|`
/// radians, derived from the model's buffers rather than tuned. That term is
/// ~2e-7 at d=256 and ~1.2e-4 at d=512 (torch's own pow rounding at three
/// pairs). The device-side `pow` this replaced was an ulp off at most pairs:
/// ~5e-4 here, past both bounds.
#[test]
fn qkv_columns_norm_rope_far_positions() {
    with_gpu(|rt| {
        for (name, d, theta) in [("qkv256_far", 256u32, 1e4f32), ("qkv512_far", 512, 1e6)] {
            let torch = load(&format!("inv_freq_{d}")).1;
            let ours = nn::rope_inv_freq(d / 2, d, theta);
            let gap = ours
                .iter()
                .zip(&torch)
                .map(|(&o, &t)| (f64::from(o) - t).abs())
                .fold(0.0, f64::max);
            let (qs, _) = load(&format!("{name}_q_in"));
            let last = load(&format!("{name}_meta")).1[0] + qs[1] as f64;
            check_qkv(rt, name, KERNEL_REL + last * gap);
        }
    });
}

#[test]
fn ple_composition_matches_the_fixtures() {
    // The model's per-layer input and PLE block from tessl's own GEMM and nn
    // kernels, exactly as `EmbedGemma2Model::encode` sequences them.
    with_gpu(|rt| {
        let (es, embeds) = load("ple_embeds");
        let (pws, proj_w) = load("ple_proj_w");
        let (rows, h) = (es[0] * es[1], es[2]);
        let n_layers = pws[0] / h;
        let li = load("ple_layer").1[0] as usize;
        let pack = |w: &[f64], out_f: usize, in_f: usize| {
            let packed = qwen35::pack_linear_weights_f32(&[&f32s(w)], &[out_f], in_f).unwrap();
            tensor_f32(rt, &[in_f, out_f], &packed)
        };
        let e = tensor_f32(rt, &[rows, h], &f32s(&embeds));
        let hidden = load("ple_hidden").1;
        let resid = tensor_f32(rt, &[rows, h], &f32s(&hidden));
        let ple = tensor_f32(rt, &[rows, h], &vec![0.0; rows * h]);
        let ple_n = tensor_f32(rt, &[rows, h], &vec![0.0; rows * h]);
        let g = tensor_f32(rt, &[rows, h], &vec![0.0; rows * h]);
        let y = tensor_f32(rt, &[rows, h], &vec![0.0; rows * h]);
        let norm_w = buf_f32(rt, &f32s(&load("ple_proj_norm_w").1));
        let slice = &proj_w[li * h * h..(li + 1) * h * h];
        gemm(&e, &pack(slice, h, h), &ple, GemmBackend::TensorOps).unwrap();
        scale_f32_inplace(rt, &ple.buffer, (h as f32).powf(-0.5), (rows * h) as u32).unwrap();
        rms_norm_f32(rt, &ple.buffer, &norm_w, &ple_n.buffer, rows as u32, h as u32, 1e-6).unwrap();
        rt.synchronize().unwrap();
        let got_pl: Vec<f64> = ple_n.buffer.read_f32()[..rows * h]
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        let want_all = load("ple_per_layer").1;
        let want_pl: Vec<f64> = (0..rows)
            .flat_map(|r| want_all[(r * n_layers + li) * h..(r * n_layers + li + 1) * h].to_vec())
            .collect();
        assert_close(
            "per_layer slice",
            &got_pl,
            &want_pl,
            &(0..want_pl.len()).collect::<Vec<_>>(),
            5e-5,
        );

        gemm(&resid, &pack(&load("ple_gate_w").1, h, h), &g, GemmBackend::TensorOps).unwrap();
        mlp_gelu_tanh(rt, &g.buffer, &ple_n.buffer, &ple.buffer, (rows * h) as u32).unwrap();
        gemm(&ple, &pack(&load("ple_out_proj_w").1, h, h), &y, GemmBackend::TensorOps).unwrap();
        let post_w = buf_f32(rt, &f32s(&load("ple_post_norm_w").1));
        rms_norm_residual_add_f32(rt, &y.buffer, &post_w, &resid.buffer, rows as u32, h as u32, 1e-6, 1.0).unwrap();
        rt.synchronize().unwrap();
        let got: Vec<f64> = resid.buffer.read_f32()[..rows * h]
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        let want = load("ple_block_out").1;
        assert_close("ple block", &got, &want, &(0..want.len()).collect::<Vec<_>>(), 5e-5);
    });
}

#[test]
fn pool_and_normalize_match_the_fixtures() {
    with_gpu(|rt| {
        let (xs, x) = load("pool_x");
        let (b, t, d) = (xs[0], xs[1], xs[2]);
        let lens: Vec<u32> = load("pool_lens").1.iter().map(|&l| l as u32).collect();
        let xb = buf_f32(rt, &f32s(&x));
        // Sequence b's live rows of the flattened [b * t, d] input.
        let segs: Vec<(u32, u32)> = lens
            .iter()
            .enumerate()
            .map(|(i, &l)| ((i * t) as u32, (i * t) as u32 + l))
            .collect();
        let sb = upload_segments(rt, &segs, (b * t) as u32).unwrap();
        let out = rt.alloc_buffer(b * d * 4).unwrap();
        segment_mean_rows(rt, &xb, &sb, &out, b as u32, (b * t) as u32, d as u32).unwrap();
        l2_normalize_rows(rt, &out, b as u32, d as u32, d as u32).unwrap();
        rt.synchronize().unwrap();
        let got: Vec<f64> = out.read_f32()[..b * d].iter().map(|&v| f64::from(v)).collect();
        let want = load("pool_out").1;
        assert_close("pool", &got, &want, &(0..want.len()).collect::<Vec<_>>(), 5e-6);

        // Matryoshka: normalizing a prefix in place equals the f64 normalize
        // of that prefix (sentence-transformers' truncate_dim with
        // normalize_embeddings; the generator's pool_out_trunc{d}), and
        // leaves the columns past it alone.
        let before = out.read_f32()[..b * d].to_vec();
        for p in [512usize, 256, 128] {
            let want = load(&format!("pool_out_trunc{p}")).1;
            let xb = buf_f32(rt, &before);
            l2_normalize_rows(rt, &xb, b as u32, p as u32, d as u32).unwrap();
            rt.synchronize().unwrap();
            let after = xb.read_f32()[..b * d].to_vec();
            let mut got = Vec::with_capacity(b * p);
            for r in 0..b {
                let row: Vec<f64> = before[r * d..r * d + p].iter().map(|&v| f64::from(v)).collect();
                let host = l2_normalize(&row);
                for j in 0..d {
                    let (g, w) = (
                        f64::from(after[r * d + j]),
                        if j < p { host[j] } else { f64::from(before[r * d + j]) },
                    );
                    assert!((g - w).abs() <= 1e-6, "prefix {p} row {r} col {j}: {g} vs {w}");
                }
                got.extend(after[r * d..r * d + p].iter().map(|&v| f64::from(v)));
            }
            assert_close(
                &format!("pool prefix {p}"),
                &got,
                &want,
                &(0..want.len()).collect::<Vec<_>>(),
                5e-6,
            );
        }
    });
}

#[test]
fn pool_and_normalize_refuse_bad_shapes() {
    with_gpu(|rt| {
        let x = rt.alloc_buffer(2 * 4 * 8 * 4).unwrap();
        let segs = buf_u32(rt, &[0, 4, 4, 8]);
        let out = rt.alloc_buffer(2 * 8 * 4).unwrap();
        assert!(segment_mean_rows(rt, &x, &segs, &out, 2, 9, 8).is_err(), "x too small");
        assert!(
            segment_mean_rows(rt, &x, &segs, &out, 3, 8, 8).is_err(),
            "segments too small"
        );
        assert!(segment_mean_rows(rt, &x, &segs, &x, 2, 8, 8).is_err(), "out aliasing x");
        assert!(
            segment_mean_rows(rt, &x, &segs, &segs, 2, 8, 8).is_err(),
            "out aliasing segments"
        );
        assert!(upload_segments(rt, &[(0, 4), (4, 9)], 8).is_err(), "end past rows");
        assert!(upload_segments(rt, &[(3, 3)], 8).is_err(), "empty range");
        assert!(upload_segments(rt, &[(5, 2)], 8).is_err(), "reversed range");
        assert!(l2_normalize_rows(rt, &out, 2, 0, 8).is_err(), "dim 0");
        assert!(l2_normalize_rows(rt, &out, 2, 9, 8).is_err(), "dim past ld");
        assert!(l2_normalize_rows(rt, &out, 3, 8, 8).is_err(), "rows past the buffer");
        // A zero row stays zero (the 1e-12 floor), not NaN.
        out.write_f32(&[0.0; 16]);
        l2_normalize_rows(rt, &out, 2, 8, 8).unwrap();
        rt.synchronize().unwrap();
        assert!(out.read_f32()[..16].iter().all(|&v| v == 0.0));
    });
}

/// The ranges are device values, so the kernel must survive any of them:
/// ends past `rows` clamp, empty and reversed ranges give zeros, and
/// overlapping, unordered ranges (rsi-jev's option spans) each get their own
/// mean, against an f64 host mean.
#[test]
fn segment_means_take_any_device_ranges() {
    with_gpu(|rt| {
        let (rows, d) = (37usize, 33usize);
        let x: Vec<f32> = (0..rows * d).map(|i| ((i * 7919) % 1013) as f32 / 97.0 - 5.0).collect();
        let xb = buf_f32(rt, &x);
        let ranges: [(u32, u32); 8] = [
            (0, 37),
            (30, 31),
            (5, 20),
            (10, 15),
            (36, 1000),
            (12, 12),
            (20, 3),
            (u32::MAX, u32::MAX),
        ];
        let flat: Vec<u32> = ranges.iter().flat_map(|&(a, b)| [a, b]).collect();
        let sb = buf_u32(rt, &flat);
        let out = rt.alloc_buffer(ranges.len() * d * 4).unwrap();
        segment_mean_rows(rt, &xb, &sb, &out, ranges.len() as u32, rows as u32, d as u32).unwrap();
        rt.synchronize().unwrap();
        let got = out.read_f32()[..ranges.len() * d].to_vec();
        for (s, &(a, b)) in ranges.iter().enumerate() {
            let end = (b as usize).min(rows);
            let start = (a as usize).min(end);
            for j in 0..d {
                let g = got[s * d + j];
                if start == end {
                    assert_eq!(g.to_bits(), 0, "segment {s} [{a}, {b}) col {j}: empty must be +0");
                    continue;
                }
                let want = (start..end).map(|r| f64::from(x[r * d + j])).sum::<f64>() / (end - start) as f64;
                assert!(
                    (f64::from(g) - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "segment {s} col {j}: {g} vs {want}"
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const REAL_TEXT_CONFIG: &str = r#"{"architectures":["EmbeddingGemma2Model"],"model_type":"embedding_gemma2","text_config":{
"attention_bias":false,"embedding_dim":768,"head_dim":256,"hidden_activation":"gelu_pytorch_tanh","hidden_size":512,
"hidden_size_per_layer_input":512,"intermediate_size":2048,
"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
"max_position_embeddings":262144,"model_type":"embedding_gemma2_text","num_attention_heads":4,"num_hidden_layers":24,
"num_key_value_heads":2,"pad_token_id":0,
"per_layer_config":{"05":{"head_dim":512,"num_key_value_heads":1},"11":{"head_dim":512,"num_key_value_heads":1},
"17":{"head_dim":512,"num_key_value_heads":1},"23":{"head_dim":512,"num_key_value_heads":1}},
"rms_norm_eps":1e-06,"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},"sliding_window":512,"vocab_size":262144}}"#;

#[test]
fn config_reads_the_checkpoint() {
    let c = EmbedGemma2Config::from_config_json(REAL_TEXT_CONFIG).unwrap();
    assert_eq!(
        (c.hidden, c.ple_dim, c.intermediate, c.vocab, c.embedding_dim),
        (512, 512, 2048, 262144, 768)
    );
    assert_eq!((c.q_heads, c.sliding_window, c.layers.len()), (4, 512, 24));
    for (i, l) in c.layers.iter().enumerate() {
        let full = i % 6 == 5;
        assert_eq!(l.sliding, !full, "layer {i}");
        assert_eq!(
            (l.head_dim, l.kv_heads),
            if full { (512, 1) } else { (256, 2) },
            "layer {i}"
        );
        assert_eq!(l.rope_theta, if full { 1e6 } else { 1e4 }, "layer {i}");
    }
}

// ---------------------------------------------------------------------------
// 7. The fused row passes and the bf16-output attention (G7)
//
// Each fused kernel is the composition of the kernels it replaced, and the
// forward's exact-f32 result must not move a bit, so the bound is equality:
// every output element is compared by its bit pattern with the unfused kernels
// run on the same inputs, over widths that are and are not powers of two (1,
// 33, 1000, 768), rows from 1 up, and a layer scale of exactly 1 (the plain
// add) and not. Outputs are poisoned with NaN first, so an element a kernel
// does not write cannot match.
//
// This block sits above `config_refuses_what_the_forward_does_not_implement`,
// not at the end of the file: that test's string literals leave its braces
// unbalanced, so `tests/hosted_selection.rs` reads it up to the next
// `#[test]`, and a helper mentioning `GpuRuntime` in between would make the
// device-free hosted test look like a device one.
// ---------------------------------------------------------------------------

use tessl::embedgemma2::{
    encoder_attn_bf16_out, resid_add_bf16_copy, resid_add_norm_bf16, resid_add_norm_f32, scale_rms_norm_inplace,
};
use tessl::nn::rms_norm_bf16;

const FUSED_EPS: f32 = 1e-6;
/// `(rows, dim)`: width 1 and 33 (a ragged simdgroup), 1000 (not a power of
/// two), the checkpoint's 512 and the 768 of its vision tower, and 2048 and
/// 4096, where the threadgroup is as wide as the launch allows.
const FUSED_SHAPES: [(u32, u32); 9] = [
    (1, 1),
    (1, 8),
    (9, 33),
    (3, 64),
    (5, 512),
    (4, 1000),
    (7, 768),
    (2, 2048),
    (3, 4096),
];
/// Layer scales: exactly 1 (the plain add) and two that are not.
const FUSED_LAYER_SCALES: [f32; 3] = [1.0, 0.37, 1.75];

fn rand32(n: usize, seed: u64, scale: f64) -> Vec<f32> {
    f32s(&random(n, seed, scale))
}

/// A checkpoint-like norm weight: around 1, not all equal.
fn norm_weight(dim: usize, seed: u64) -> Vec<f32> {
    random(dim, seed, 0.8).iter().map(|&x| (1.0 + x) as f32).collect()
}

fn bits32(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn poisoned_f32(rt: &Arc<GpuRuntime>, n: usize) -> GpuBuffer {
    buf_f32(rt, &vec![f32::NAN; n])
}

/// A bf16 buffer of `n` elements filled with the NaN pattern.
fn poisoned_bf16(rt: &Arc<GpuRuntime>, n: usize) -> GpuBuffer {
    let b = rt.alloc_buffer(n.max(1) * 2).unwrap();
    b.write_bf16_bits(&vec![0x7fc0u16; n.max(1)]);
    b
}

fn read_bf16_bits(rt: &Arc<GpuRuntime>, b: &GpuBuffer, n: usize) -> Vec<u16> {
    rt.synchronize().unwrap();
    b.try_contents_u16().unwrap()[..n].to_vec()
}

/// `bf16_rne(x)` as the 16 stored bits.
fn bf16_bits_of(x: f32) -> u16 {
    (bf16_rne(x).to_bits() >> 16) as u16
}

#[test]
fn config_refuses_what_the_forward_does_not_implement() {
    let cases = [
        ("\"gelu_pytorch_tanh\"", "\"silu\""),
        ("\"head_dim\":256", "\"head_dim\":128"),
        (
            "\"05\":{\"head_dim\":512,",
            "\"05\":{\"sliding_window\":3,\"head_dim\":512,",
        ),
        ("\"05\":{", "\"99\":{"),
        ("\"num_key_value_heads\":2", "\"num_key_value_heads\":3"),
        ("\"attention_bias\":false", "\"attention_bias\":true"),
        (
            "\"rope_theta\":10000.0,\"rope_type\":\"default\"",
            "\"rope_theta\":10000.0,\"rope_type\":\"yarn\"",
        ),
        ("\"num_hidden_layers\":24", "\"num_hidden_layers\":23"),
        ("\"sliding_window\":512", "\"sliding_window\":0"),
        (
            "\"model_type\":\"embedding_gemma2_text\"",
            "\"model_type\":\"gemma4_text\"",
        ),
    ];
    for (from, to) in cases {
        assert!(REAL_TEXT_CONFIG.contains(from), "test bug: {from} not in the config");
        let text = REAL_TEXT_CONFIG.replacen(from, to, 1);
        assert!(
            EmbedGemma2Config::from_config_json(&text).is_err(),
            "{from} -> {to}: accepted"
        );
    }
}

#[test]
fn the_fused_row_passes_launch_as_wide_as_the_kernels_they_replace() {
    // The fused passes reduce a row in the lane order the unfused norms do only
    // if both launch the same number of lanes per row; the host picks that from
    // each pipeline's limit (`nn::reduce_tptg`), so the limits must agree.
    use objc2_metal::MTLComputePipelineState;
    with_gpu(|rt| {
        let max = |name: &str| rt.pipeline(name).unwrap().maxTotalThreadsPerThreadgroup().min(1024);
        for (fused, unfused) in [
            ("eg2t_resid_norm_f32", "rms_norm_residual_add_f32"),
            ("eg2t_resid_norm_f32", "rms_norm_f32"),
            ("eg2t_resid_norm_bf16", "rms_norm_bf16"),
            ("eg2t_resid_add_bf16_copy", "rms_norm_residual_add_f32"),
            ("eg2t_scale_rms_norm_f32", "rms_norm_f32"),
        ] {
            assert_eq!(max(fused), max(unfused), "{fused} against {unfused}");
        }
    });
}

#[test]
fn resid_add_norm_f32_is_the_unfused_pair_bit_for_bit() {
    with_gpu(|rt| {
        for (rows, dim) in FUSED_SHAPES {
            for s in FUSED_LAYER_SCALES {
                let n = (rows * dim) as usize;
                let seed = 700 + u64::from(rows) * 31 + u64::from(dim);
                let y = buf_f32(rt, &rand32(n, seed, 3.0));
                let resid0 = rand32(n, seed + 1, 5.0);
                let w_post = buf_f32(rt, &norm_weight(dim as usize, seed + 2));
                let w_next = buf_f32(rt, &norm_weight(dim as usize, seed + 3));
                let (resid_u, out_u) = (buf_f32(rt, &resid0), poisoned_f32(rt, n));
                rms_norm_residual_add_f32(rt, &y, &w_post, &resid_u, rows, dim, FUSED_EPS, s).unwrap();
                rms_norm_f32(rt, &resid_u, &w_next, &out_u, rows, dim, FUSED_EPS).unwrap();
                let (resid_f, out_f) = (buf_f32(rt, &resid0), poisoned_f32(rt, n));
                resid_add_norm_f32(rt, &y, &w_post, &resid_f, &w_next, &out_f, rows, dim, FUSED_EPS, s).unwrap();
                rt.synchronize().unwrap();
                let what = format!("{rows}x{dim} layer_scale {s}");
                assert_eq!(
                    bits32(&resid_f.read_f32()[..n]),
                    bits32(&resid_u.read_f32()[..n]),
                    "residual, {what}"
                );
                assert_eq!(
                    bits32(&out_f.read_f32()[..n]),
                    bits32(&out_u.read_f32()[..n]),
                    "normed output, {what}"
                );
            }
        }
    });
}

#[test]
fn resid_add_norm_bf16_is_the_unfused_pair_bit_for_bit() {
    with_gpu(|rt| {
        for (rows, dim) in FUSED_SHAPES {
            for s in FUSED_LAYER_SCALES {
                let n = (rows * dim) as usize;
                let seed = 900 + u64::from(rows) * 31 + u64::from(dim);
                let y = buf_f32(rt, &rand32(n, seed, 3.0));
                let resid0 = rand32(n, seed + 1, 5.0);
                let w_post = buf_f32(rt, &norm_weight(dim as usize, seed + 2));
                let w_next = buf_f32(rt, &norm_weight(dim as usize, seed + 3));
                let (resid_u, out_u) = (buf_f32(rt, &resid0), poisoned_bf16(rt, n));
                rms_norm_residual_add_f32(rt, &y, &w_post, &resid_u, rows, dim, FUSED_EPS, s).unwrap();
                rms_norm_bf16(rt, &resid_u, &w_next, &out_u, rows, dim, FUSED_EPS).unwrap();
                let (resid_f, out_f) = (buf_f32(rt, &resid0), poisoned_bf16(rt, n));
                resid_add_norm_bf16(rt, &y, &w_post, &resid_f, &w_next, &out_f, rows, dim, FUSED_EPS, s).unwrap();
                rt.synchronize().unwrap();
                let what = format!("{rows}x{dim} layer_scale {s}");
                assert_eq!(
                    bits32(&resid_f.read_f32()[..n]),
                    bits32(&resid_u.read_f32()[..n]),
                    "residual, {what}"
                );
                assert_eq!(
                    read_bf16_bits(rt, &out_f, n),
                    read_bf16_bits(rt, &out_u, n),
                    "bf16 normed output, {what}"
                );
            }
        }
    });
}

#[test]
fn resid_add_bf16_copy_is_the_unfused_add_and_cast_bit_for_bit() {
    with_gpu(|rt| {
        for (rows, dim) in FUSED_SHAPES {
            for s in FUSED_LAYER_SCALES {
                let n = (rows * dim) as usize;
                let seed = 1100 + u64::from(rows) * 31 + u64::from(dim);
                let y = buf_f32(rt, &rand32(n, seed, 3.0));
                let resid0 = rand32(n, seed + 1, 5.0);
                let w_post = buf_f32(rt, &norm_weight(dim as usize, seed + 2));
                let resid_u = buf_f32(rt, &resid0);
                rms_norm_residual_add_f32(rt, &y, &w_post, &resid_u, rows, dim, FUSED_EPS, s).unwrap();
                let (resid_f, copy_f) = (buf_f32(rt, &resid0), poisoned_bf16(rt, n));
                resid_add_bf16_copy(rt, &y, &w_post, &resid_f, &copy_f, rows, dim, FUSED_EPS, s).unwrap();
                rt.synchronize().unwrap();
                let what = format!("{rows}x{dim} layer_scale {s}");
                let want = resid_u.read_f32();
                assert_eq!(bits32(&resid_f.read_f32()[..n]), bits32(&want[..n]), "residual, {what}");
                // The cast is the GPU's round to nearest even, `bf16_rne` on the host.
                let want_bits: Vec<u16> = want[..n].iter().map(|&x| bf16_bits_of(x)).collect();
                assert_eq!(read_bf16_bits(rt, &copy_f, n), want_bits, "bf16 copy, {what}");
            }
        }
    });
}

#[test]
fn scale_rms_norm_inplace_is_the_unfused_pair_bit_for_bit() {
    with_gpu(|rt| {
        // 512^-0.5 (the checkpoint's: not a power of two, so the scale rounds),
        // 1, and a scale that makes some rows' squares large.
        for scale in [512f32.powf(-0.5), 1.0, 3.5] {
            for (rows, dim) in FUSED_SHAPES {
                let n = (rows * dim) as usize;
                let seed = 1300 + u64::from(rows) * 31 + u64::from(dim);
                let x0 = rand32(n, seed, 4.0);
                let w = buf_f32(rt, &norm_weight(dim as usize, seed + 1));
                let xu = buf_f32(rt, &x0);
                scale_f32_inplace(rt, &xu, scale, rows * dim).unwrap();
                rms_norm_f32(rt, &xu, &w, &xu, rows, dim, FUSED_EPS).unwrap();
                let xf = buf_f32(rt, &x0);
                scale_rms_norm_inplace(rt, &xf, &w, rows, dim, FUSED_EPS, scale).unwrap();
                rt.synchronize().unwrap();
                assert_eq!(
                    bits32(&xf.read_f32()[..n]),
                    bits32(&xu.read_f32()[..n]),
                    "{rows}x{dim} scale {scale}"
                );
            }
        }
    });
}

#[test]
fn the_fused_row_passes_refuse_what_they_cannot_run() {
    with_gpu(|rt| {
        let (rows, dim) = (4u32, 64u32);
        let n = (rows * dim) as usize;
        let (y, resid) = (buf_f32(rt, &rand32(n, 1, 1.0)), buf_f32(rt, &rand32(n, 2, 1.0)));
        let (w1, w2) = (buf_f32(rt, &norm_weight(64, 3)), buf_f32(rt, &norm_weight(64, 4)));
        let out = poisoned_f32(rt, n);
        resid_add_norm_f32(rt, &y, &w1, &resid, &w2, &out, rows, dim, FUSED_EPS, 1.0).expect("the valid call");
        rt.synchronize().unwrap();
        let before = out.read_f32();
        let short_w = buf_f32(rt, &norm_weight(63, 5));
        let (bf_out, bf_short) = (poisoned_bf16(rt, n), poisoned_bf16(rt, n - 1));
        let results = [
            (
                "dim 0",
                resid_add_norm_f32(rt, &y, &w1, &resid, &w2, &out, rows, 0, FUSED_EPS, 1.0),
            ),
            (
                "eps 0",
                resid_add_norm_f32(rt, &y, &w1, &resid, &w2, &out, rows, dim, 0.0, 1.0),
            ),
            (
                "a NaN layer scale",
                resid_add_norm_f32(rt, &y, &w1, &resid, &w2, &out, rows, dim, FUSED_EPS, f32::NAN),
            ),
            (
                "out aliasing resid",
                resid_add_norm_f32(rt, &y, &w1, &resid, &w2, &resid, rows, dim, FUSED_EPS, 1.0),
            ),
            (
                "a weight one element short",
                resid_add_norm_f32(rt, &y, &short_w, &resid, &w2, &out, rows, dim, FUSED_EPS, 1.0),
            ),
            (
                "an f32 out given bf16 storage",
                resid_add_norm_f32(rt, &y, &w1, &resid, &w2, &bf_out, rows, dim, FUSED_EPS, 1.0),
            ),
            (
                "a bf16 out one element short",
                resid_add_norm_bf16(rt, &y, &w1, &resid, &w2, &bf_short, rows, dim, FUSED_EPS, 1.0),
            ),
            (
                "a bf16 copy one element short",
                resid_add_bf16_copy(rt, &y, &w1, &resid, &bf_short, rows, dim, FUSED_EPS, 1.0),
            ),
            (
                "scale_rms_norm with an infinite scale",
                scale_rms_norm_inplace(rt, &resid, &w1, rows, dim, FUSED_EPS, f32::INFINITY),
            ),
            (
                "scale_rms_norm with the weight aliasing x",
                scale_rms_norm_inplace(rt, &w1, &w1, 1, dim, FUSED_EPS, 1.0),
            ),
        ];
        for (what, r) in results {
            assert!(r.is_err(), "{what}: accepted");
        }
        // A refused call writes nothing.
        rt.synchronize().unwrap();
        assert_eq!(
            bits32(&out.read_f32()[..n]),
            bits32(&before[..n]),
            "a refusal wrote to out"
        );
        // Zero rows is a no-op, not an error.
        resid_add_norm_f32(rt, &y, &w1, &resid, &w2, &out, 0, dim, FUSED_EPS, 1.0).expect("0 rows");
        scale_rms_norm_inplace(rt, &resid, &w1, 0, dim, FUSED_EPS, 1.0).expect("0 rows");
    });
}

/// `bf16o` (bf16 operands, bf16 output) against `TiledBf16` (f32 output) on
/// the same bf16 operands: the output is the f32 result rounded to nearest
/// even, every element, padding rows as +0, at the text encoder's head dims
/// (256, 512) and the vision tower's (64), with a window, ragged lengths and a
/// length past `seq`.
#[test]
fn the_bf16_output_attention_is_the_bf16_kernel_rounded_bit_for_bit() {
    with_gpu(|rt| {
        let cases = [
            // (batch, seq, heads, heads_kv, head_dim, window, lens)
            (3usize, 37usize, 4usize, 2usize, 256usize, 3usize, vec![37usize, 20, 1]),
            (2, 70, 4, 1, 512, 0, vec![70, 33]),
            (2, 40, 6, 2, 64, 5, vec![40, 17]),
            (1, 129, 4, 2, 256, 0, vec![129]),
        ];
        for (ci, (b, t, h, hkv, d, window, lens)) in cases.into_iter().enumerate() {
            let a = Attn {
                b,
                t,
                h,
                hkv,
                d,
                window,
                lens,
            };
            let q = rand32(b * t * h * d, 40 + ci as u64, 0.25);
            let k = rand32(b * t * hkv * d, 50 + ci as u64, 0.25);
            let v = rand32(b * t * hkv * d, 60 + ci as u64, 1.0);
            let lens: Vec<u32> = a.lens.iter().map(|&l| l as u32).collect();
            // The f32-output kernel on these operands, as the unfused forward ran it.
            let f32_out = run_attn(rt, EncoderAttnKernel::TiledBf16, &a, &q, &k, &v, &lens);
            // The bf16-output kernel on the very same operands.
            let n = q.len();
            let o = poisoned_bf16(rt, n);
            encoder_attn_bf16_out(
                rt,
                &bf16_buf(rt, &q),
                &bf16_buf(rt, &k),
                &bf16_buf(rt, &v),
                &o,
                &buf_u32(rt, &lens),
                EncoderAttnDims {
                    batch: b as u32,
                    seq: t as u32,
                    heads: h as u32,
                    heads_kv: hkv as u32,
                    head_dim: d as u32,
                    window: window as u32,
                    scale: 1.0,
                },
            )
            .unwrap_or_else(|e| panic!("D={d}: {e}"));
            let got = read_bf16_bits(rt, &o, n);
            let want: Vec<u16> = f32_out.iter().map(|&x| bf16_bits_of(x as f32)).collect();
            let wrong = got.iter().zip(&want).position(|(g, w)| g != w);
            assert!(
                wrong.is_none(),
                "D={d} window={window}: element {:?} is {:#06x}, the rounded f32-output kernel gives {:#06x}",
                wrong,
                wrong.map_or(0, |i| got[i]),
                wrong.map_or(0, |i| want[i])
            );
            // The padding rows are +0 in bf16 too (bits 0).
            for bi in 0..b {
                for ti in a.lens[bi].min(t)..t {
                    let base = (bi * t + ti) * h * d;
                    assert!(
                        got[base..base + h * d].iter().all(|&x| x == 0),
                        "D={d}: padding row b={bi} t={ti} is not +0"
                    );
                }
            }
        }
        // Refusals: head dim 32 has no bf16o kernel, and `o` must hold bf16
        // for every element.
        let dims = EncoderAttnDims {
            batch: 1,
            seq: 8,
            heads: 2,
            heads_kv: 1,
            head_dim: 64,
            window: 0,
            scale: 1.0,
        };
        let lens = buf_u32(rt, &[8]);
        let (q, k, v) = (
            rt.alloc_buffer(8 * 2 * 64 * 2).unwrap(),
            rt.alloc_buffer(8 * 64 * 2).unwrap(),
            rt.alloc_buffer(8 * 64 * 2).unwrap(),
        );
        let o = rt.alloc_buffer(8 * 2 * 64 * 2).unwrap();
        encoder_attn_bf16_out(rt, &q, &k, &v, &o, &lens, dims).expect("the valid call");
        let short = rt.alloc_buffer(8 * 2 * 64 * 2 - 2).unwrap();
        assert!(
            encoder_attn_bf16_out(rt, &q, &k, &v, &short, &lens, dims).is_err(),
            "a bf16 o one element short: accepted"
        );
        let d32 = EncoderAttnDims { head_dim: 32, ..dims };
        assert!(
            encoder_attn_bf16_out(rt, &q, &k, &v, &o, &lens, d32).is_err(),
            "the bf16 output kernel at head dim 32: accepted"
        );
    });
}
