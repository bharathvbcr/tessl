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
    encoder_attn, l2_normalize_rows, segment_mean_rows, upload_segments, EmbedGemma2Config, EncoderAttnDims,
};
use tessl::gemm::{gemm, GemmBackend};
use tessl::nn::{self, mlp_gelu_tanh, rms_norm_f32, rms_norm_residual_add_f32, scale_f32_inplace};
use tessl::npy::read_npy;
use tessl::qwen35::{self, AttnShape, AttnTargets, Cols, QkvColumns};
use tessl::tensor::GpuBuffer;
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

fn run_attn(rt: &Arc<GpuRuntime>, a: &Attn, q: &[f32], k: &[f32], v: &[f32], lens: &[u32]) -> Vec<f64> {
    let (qb, kb, vb) = (buf_f32(rt, q), buf_f32(rt, k), buf_f32(rt, v));
    let o = rt.alloc_buffer(q.len().max(1) * 4).unwrap();
    // Poison the output so an unwritten element cannot pass as zero.
    o.write_f32(&vec![f32::NAN; q.len()]);
    let lb = buf_u32(rt, lens);
    encoder_attn(
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
        false,
    )
    .unwrap();
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
        for name in ["swa", "global", "swa_tiny"] {
            let (a, q, k, v, want) = attn_fixture(name);
            let lens: Vec<u32> = a.lens.iter().map(|&l| l as u32).collect();
            let got = run_attn(rt, &a, &f32s(&q), &f32s(&k), &f32s(&v), &lens);
            assert_close(name, &got, &want, &valid_rows(&a), KERNEL_REL);
            padding_is_zero(name, &a, &got);
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

fn check_random(rt: &Arc<GpuRuntime>, a: &Attn, seed: u64) {
    let q = random(a.b * a.t * a.h * a.d, seed, 0.25);
    let k = random(a.b * a.t * a.hkv * a.d, seed + 1, 0.25);
    let v = random(a.b * a.t * a.hkv * a.d, seed + 2, 1.0);
    let lens: Vec<u32> = a.lens.iter().map(|&l| l as u32).collect();
    let got = run_attn(rt, a, &f32s(&q), &f32s(&k), &f32s(&v), &lens);
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
        &format!("random D={} window={} lens={:?}", a.d, a.window, a.lens),
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
        // Sliding layers: D=256, 4 query heads over 2 KV heads, window 512,
        // one sequence longer than the window on both sides, ragged others.
        check_random(
            rt,
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
    });
}

#[test]
fn encoder_attn_row_matches_the_same_sequence_alone_bit_for_bit() {
    with_gpu(|rt| {
        let (h, hkv, d, window) = (4usize, 2usize, 256usize, 512usize);
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
        let batched = run_attn(rt, &batched_attn, &qb, &kb, &vb, &[t_long as u32, len as u32]);
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
            &alone_attn,
            &q[..len * h * d],
            &k[..len * hkv * d],
            &v[..len * hkv * d],
            &[len as u32],
        );
        let row1 = &batched[t_long * h * d..t_long * h * d + len * h * d];
        for (i, (x, y)) in row1.iter().zip(&alone).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i}: batched {x} vs alone {y}");
        }
    });
}

#[test]
fn encoder_attn_is_bitwise_repeatable() {
    with_gpu(|rt| {
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
        let first = run_attn(rt, &a, &q, &k, &v, &[300, 120]);
        let second = run_attn(rt, &a, &q, &k, &v, &[300, 120]);
        assert!(first.iter().zip(&second).all(|(x, y)| x.to_bits() == y.to_bits()));
    });
}

#[test]
fn encoder_attn_clamps_device_lengths() {
    // The host cannot see a device length: 0 gives an all-zero row, and one
    // past the padded length behaves as the padded length.
    with_gpu(|rt| {
        let a = Attn {
            b: 2,
            t: 40,
            h: 2,
            hkv: 1,
            d: 256,
            window: 7,
            lens: vec![40, 0],
        };
        let q = random(a.b * a.t * a.h * a.d, 41, 0.25);
        let k = random(a.b * a.t * a.hkv * a.d, 42, 0.25);
        let v = random(a.b * a.t * a.hkv * a.d, 43, 1.0);
        let got = run_attn(rt, &a, &f32s(&q), &f32s(&k), &f32s(&v), &[u32::MAX, 0]);
        let want = attn_ref(&Attn { lens: vec![40, 0], ..a }, &q, &k, &v, None);
        let idx: Vec<usize> = (0..got.len()).collect();
        assert_close("clamped", &got, &want, &idx, KERNEL_REL);
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
        for (scale, window) in [(0.25, 0usize), (4.0, 0), (0.25, 3), (4.0, 3)] {
            let a = Attn {
                b: 4,
                t: 12,
                h: 2,
                hkv: 1,
                d: 256,
                window,
                lens: vec![5, 0, 12, 1],
            };
            let q = random(a.b * a.t * a.h * a.d, 51, scale);
            let k = random(a.b * a.t * a.hkv * a.d, 52, scale);
            let v = random(a.b * a.t * a.hkv * a.d, 53, 1.0);
            let lens: Vec<u32> = a.lens.iter().map(|&l| l as u32).collect();
            let got = run_attn(rt, &a, &f32s(&q), &f32s(&k), &f32s(&v), &lens);
            let what = format!("scale={scale} window={window}");
            assert!(got.iter().all(|x| x.is_finite()), "{what}: non-finite output");
            padding_is_zero(&what, &a, &got);
            let want = attn_ref(&a, &q, &k, &v, None);
            assert_close(&what, &got, &want, &valid_rows(&a), KERNEL_REL);
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
        encoder_attn(rt, &q, &k, &v, &o, &lens, ok, false).expect("the baseline call");
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
                encoder_attn(rt, &q, &k, &v, &o, &lens, dims, false).is_err(),
                "{what}: accepted"
            );
        }
        assert!(
            encoder_attn(rt, &q, &k, &v, &q, &lens, ok, false).is_err(),
            "o aliasing q: accepted"
        );
        let short_lens = buf_u32(rt, &[8]);
        assert!(
            encoder_attn(rt, &q, &k, &v, &o, &short_lens, ok, false).is_err(),
            "short lens: accepted"
        );
    });
}

/// Run the columns norm+RoPE on fixture `name` at its own positions (the
/// caches hold `pos0 + T` slots, so the kernel writes at the fixture's real
/// positions) and compare q/k/v with the fixture within `rel`.
fn check_qkv(rt: &Arc<GpuRuntime>, name: &str, rel: f64) {
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
    let qo = rt.alloc_buffer(t * hq * d * 4).unwrap();
    let ko = rt.alloc_buffer(cap * hkv * d * 4).unwrap();
    let vo = rt.alloc_buffer(cap * hkv * d * 4).unwrap();
    let vn = rt.alloc_buffer(cap * hkv * d * 4).unwrap();
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
        },
        0.0,
        &qw,
        &kw,
        &AttnTargets {
            q_out: &qo,
            k_cache: &ko,
            v_cache: &vo,
        },
        pos0 as u32,
        theta,
        1e-6,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    let ones = buf_f32(rt, &vec![1.0; d]);
    rms_norm_f32(rt, &vo, &ones, &vn, (cap * hkv) as u32, d as u32, 1e-6).unwrap();
    rt.synchronize().unwrap();
    let slots = |b: &GpuBuffer| -> Vec<f64> {
        b.read_f32()[pos0 * hkv * d..cap * hkv * d]
            .iter()
            .map(|&x| f64::from(x))
            .collect()
    };
    let got_q: Vec<f64> = qo.read_f32()[..t * hq * d].iter().map(|&x| f64::from(x)).collect();
    for (what, got, want) in [
        ("q", got_q, load(&format!("{name}_q_out")).1),
        ("k", slots(&ko), load(&format!("{name}_k_out")).1),
        ("v", slots(&vn), load(&format!("{name}_v_out")).1),
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
        assert!(differ <= want.len() / 32, "d={d}: {differ} of {} pairs differ from torch", want.len());
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
            let gap = ours.iter().zip(&torch).map(|(&o, &t)| (f64::from(o) - t).abs()).fold(0.0, f64::max);
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
