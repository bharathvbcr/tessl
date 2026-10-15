//! EmbeddingGemma 2's audio kernels against f64 references, at small shapes
//! and at the real checkpoint's.
//!
//! The references are `tests/common/embedgemma2_audio.rs`: the chunked
//! attention is written there the way transformers writes it (blocked q/k/v,
//! `_rel_shift` by pad/flatten/view, the 4-D mask converted to the 5-D blocked
//! layout), and `tests/embedgemma2_audio_fixtures.rs` holds that reference to
//! transformers' own outputs. The kernels are then held to it at the fixtures'
//! sizes and at the real ones (128 mel bins and 128 channels in the first conv,
//! hidden 1024 and 1536, 8 heads of 128, chunk 12, window 12), over ragged
//! batches of clips with trailing, interior and total padding.
//!
//! Bound, fixed before any run: an f32 kernel against an f64 reference within
//! `2e-5 * max|ref|` (one f32 rounding per step over at most a few thousand
//! terms), the same as `tests/embedgemma2_kernels.rs`.

mod common;

use std::sync::Arc;

use common::embedgemma2_audio::{
    attention_ref, clamp, conv_layer, q_mult, relative_keys, rms_eps, sigmoid, silu, AttnSpec,
};
use common::{buf, buf_u32, seeded, with_gpu, SplitMix};
use tessl::embedgemma2_audio::{
    attention, dwconv1d_causal, ew, glu, rms_norm, sub_conv0_ln_relu, sub_conv1_ln_relu, AttentionSpec, Clamp,
    ClipTable, Ew, EwAdd, RmsNorm,
};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

const KERNEL_REL: f64 = 2e-5;
/// A value no kernel writes: an output element still holding it was skipped.
const UNWRITTEN: f32 = 12345.0;

fn rand(r: &mut SplitMix, n: usize, scale: f64, offset: f64) -> Vec<f32> {
    (0..n).map(|_| (offset + f64::from(r.unit()) * scale) as f32).collect()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn read(rt: &Arc<GpuRuntime>, b: &GpuBuffer, n: usize) -> Vec<f64> {
    rt.synchronize().unwrap();
    b.read_f32()[..n].iter().map(|&v| f64::from(v)).collect()
}

fn assert_close(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
    let (mut worst, mut at) = (0.0f64, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = (g - w).abs();
        if !e.is_finite() || e > worst {
            worst = if e.is_finite() { e } else { f64::INFINITY };
            at = i;
        }
    }
    eprintln!("{what}: max abs error {worst:.3e} of scale {scale:.3e}");
    assert!(
        worst <= KERNEL_REL * scale,
        "{what}: max abs error {worst:.3e} at {at} exceeds {KERNEL_REL:.0e} * {scale:.3e} (got {}, want {})",
        got[at],
        want[at]
    );
}

fn mask_u32(masks: &[Vec<bool>]) -> Vec<u32> {
    masks.iter().flatten().map(|&m| u32::from(m)).collect()
}

/// Clip masks of the given lengths with every kind of padding.
fn masks(frames: &[usize]) -> Vec<Vec<bool>> {
    frames
        .iter()
        .enumerate()
        .map(|(i, &f)| match i % 4 {
            0 => vec![true; f],
            1 => (0..f).map(|t| t < f * 3 / 4).collect(),
            2 => (0..f).map(|t| !(f / 4..f / 2).contains(&t)).collect(),
            _ => vec![false; f],
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Subsampling convolutions
// ---------------------------------------------------------------------------

fn conv0_case(rt: &Arc<GpuRuntime>, frames: &[usize], mels: usize, channels: usize, eps: f32) {
    let mut r = SplitMix::new(1 + mels as u64);
    let ms = masks(frames);
    let table = ClipTable::new(rt, frames).unwrap();
    let feats: Vec<Vec<f32>> = frames.iter().map(|&f| rand(&mut r, f * mels, 2.0, 0.0)).collect();
    let w = rand(&mut r, channels * 9, 0.5, 0.0);
    let gamma = rand(&mut r, channels, 0.3, 1.0);
    let mels1 = mels.div_ceil(2);
    let n_out = table.mid_rows * mels1 * channels;
    let out = seeded(rt, n_out, UNWRITTEN);
    sub_conv0_ln_relu(
        rt,
        &buf(rt, &feats.concat()),
        &buf_u32(rt, &mask_u32(&ms)),
        &buf(rt, &w),
        &buf(rt, &gamma),
        &out,
        &table,
        mels as u32,
        channels as u32,
        eps,
    )
    .unwrap();
    let got = read(rt, &out, n_out);
    for (i, c) in table.clips.iter().enumerate() {
        let (want, (_, to, fo), _) = conv_layer(&f64s(&feats[i]), (1, c.frames, mels), &ms[i], &f64s(&w), &f64s(&gamma), f64::from(eps));
        assert_eq!((to, fo), (c.mid_rows, mels1));
        let mut g = Vec::new();
        let mut e = Vec::new();
        for ti in 0..to {
            for fi in 0..fo {
                for ch in 0..channels {
                    g.push(got[((c.mid0 + ti) * mels1 + fi) * channels + ch]);
                    e.push(want[(ch * to + ti) * fo + fi]);
                }
            }
        }
        assert_close(&format!("conv0 mels {mels} channels {channels} clip {i} ({} frames)", c.frames), &g, &e);
    }
}

#[test]
fn sub_conv0_matches_the_f64_reference() {
    with_gpu(|rt| {
        // Small: odd sizes, a 1-frame clip, a clip with no valid frame.
        conv0_case(rt, &[7, 12, 1, 5], 9, 4, 1e-5);
        conv0_case(rt, &[3, 2], 16, 33, 1e-6);
        // The checkpoint's: 128 mel bins, 128 channels; 99 and 330 frames.
        conv0_case(rt, &[99, 330, 41, 17], 128, 128, 1e-6);
        // More channels than threads of one simdgroup, and the limit.
        conv0_case(rt, &[20, 9], 12, 512, 1e-6);
    });
}

fn conv1_case(rt: &Arc<GpuRuntime>, frames: &[usize], mels1: usize, c0: usize, c1: usize, eps: f32) {
    let mut r = SplitMix::new(7 + c0 as u64);
    let ms = masks(frames);
    let table = ClipTable::new(rt, frames).unwrap();
    // The first layer's output is post-ReLU: non-negative.
    let mid: Vec<f32> = rand(&mut r, table.mid_rows * mels1 * c0, 1.0, 0.0).iter().map(|v| v.abs()).collect();
    // The checkpoint's [c1, c0, 3, 3], and the kernel's [tap, c0, c1].
    let w_ck = rand(&mut r, c1 * c0 * 9, 2.0 / (9.0 * c0 as f64).sqrt(), 0.0);
    let mut w_k = vec![0.0f32; 9 * c0 * c1];
    for co in 0..c1 {
        for ci in 0..c0 {
            for kh in 0..3 {
                for kw in 0..3 {
                    w_k[((kh * 3 + kw) * c0 + ci) * c1 + co] = w_ck[((co * c0 + ci) * 3 + kh) * 3 + kw];
                }
            }
        }
    }
    let gamma = rand(&mut r, c1, 0.3, 1.0);
    let mels2 = mels1.div_ceil(2);
    let n_out = table.rows * mels2 * c1;
    let out = seeded(rt, n_out, UNWRITTEN);
    sub_conv1_ln_relu(
        rt,
        &buf(rt, &mid),
        &buf_u32(rt, &mask_u32(&ms)),
        &buf(rt, &w_k),
        &buf(rt, &gamma),
        &out,
        &table,
        mels1 as u32,
        c0 as u32,
        c1 as u32,
        eps,
    )
    .unwrap();
    let got = read(rt, &out, n_out);
    for (i, c) in table.clips.iter().enumerate() {
        // Channels-first input of this clip, and the mask `mask[::2]`.
        let mut x = vec![0.0; c0 * c.mid_rows * mels1];
        for ti in 0..c.mid_rows {
            for fi in 0..mels1 {
                for ci in 0..c0 {
                    x[(ci * c.mid_rows + ti) * mels1 + fi] = f64::from(mid[((c.mid0 + ti) * mels1 + fi) * c0 + ci]);
                }
            }
        }
        let m1: Vec<bool> = ms[i].iter().step_by(2).copied().collect();
        let (want, (_, t2, f2), _) = conv_layer(&x, (c0, c.mid_rows, mels1), &m1, &f64s(&w_ck), &f64s(&gamma), f64::from(eps));
        assert_eq!((t2, f2), (c.rows, mels2));
        let mut g = Vec::new();
        let mut e = Vec::new();
        for ti in 0..t2 {
            for fi in 0..f2 {
                for ch in 0..c1 {
                    g.push(got[(c.row0 + ti) * mels2 * c1 + fi * c1 + ch]);
                    e.push(want[(ch * t2 + ti) * f2 + fi]);
                }
            }
        }
        assert_close(&format!("conv1 c0 {c0} c1 {c1} clip {i} ({} frames)", c.frames), &g, &e);
    }
}

#[test]
fn sub_conv1_matches_the_f64_reference() {
    with_gpu(|rt| {
        conv1_case(rt, &[7, 12, 1, 5], 5, 4, 3, 1e-5);
        conv1_case(rt, &[9, 16], 7, 6, 40, 1e-6);
        // The checkpoint's: 64 frequency bins of 128 channels to 32.
        conv1_case(rt, &[99, 330, 41, 17], 64, 128, 32, 1e-6);
        // The widest second layer the kernel holds (four channels per lane).
        conv1_case(rt, &[10, 3], 6, 8, 128, 1e-6);
    });
}

// ---------------------------------------------------------------------------
// Norms and elementwise passes
// ---------------------------------------------------------------------------

struct NormCase {
    rows: usize,
    dim: usize,
    weight: bool,
    pre: [Clamp; 2],
    silu: bool,
    post: Clamp,
    scale: f32,
    resid: bool,
}

fn norm_case(rt: &Arc<GpuRuntime>, c: &NormCase, eps: f32, seed: u64) {
    let mut r = SplitMix::new(seed);
    let x = rand(&mut r, c.rows * c.dim, 3.0, 0.0);
    let w = rand(&mut r, c.dim, 0.3, 1.0);
    let res = rand(&mut r, c.rows * c.dim, 2.0, 0.0);
    let (xb, wb, rb) = (buf(rt, &x), buf(rt, &w), buf(rt, &res));
    let mut spec = RmsNorm::new(eps, c.weight.then_some(&wb));
    spec.pre = c.pre;
    spec.silu = c.silu;
    spec.post = c.post;
    spec.scale = c.scale;
    spec.resid = c.resid.then_some(&rb);
    let n = c.rows * c.dim;
    let out = seeded(rt, n, UNWRITTEN);
    rms_norm(rt, &xb, &out, c.rows as u32, c.dim as u32, &spec).unwrap();
    let got = read(rt, &out, n);

    let mut want = Vec::new();
    for row in 0..c.rows {
        let v: Vec<f64> = x[row * c.dim..(row + 1) * c.dim]
            .iter()
            .map(|&v| {
                let a = clamp(f64::from(v), f64::from(c.pre[0].lo), f64::from(c.pre[0].hi));
                clamp(a, f64::from(c.pre[1].lo), f64::from(c.pre[1].hi))
            })
            .collect();
        let wv = f64s(&w);
        let normed = rms_eps(&v, c.weight.then_some(wv.as_slice()), f64::from(eps));
        for (d, y) in normed.into_iter().enumerate() {
            let y = if c.silu { silu(y) } else { y };
            let y = clamp(y, f64::from(c.post.lo), f64::from(c.post.hi)) * f64::from(c.scale);
            want.push(if c.resid { y + f64::from(res[row * c.dim + d]) } else { y });
        }
    }
    assert_close(
        &format!("rms_norm {}x{} weight {} silu {} resid {}", c.rows, c.dim, c.weight, c.silu, c.resid),
        &got,
        &want,
    );

    // In place (`out` is `x`) is the out-of-place result bit for bit.
    if !c.resid {
        let ip = buf(rt, &x);
        let mut spec = RmsNorm::new(eps, c.weight.then_some(&wb));
        spec.pre = c.pre;
        spec.silu = c.silu;
        spec.post = c.post;
        spec.scale = c.scale;
        rms_norm(rt, &ip, &ip, c.rows as u32, c.dim as u32, &spec).unwrap();
        assert_eq!(read(rt, &ip, n), got, "in-place rms_norm differs from out-of-place");
    }
}

#[test]
fn rms_norm_matches_the_f64_reference() {
    let none = Clamp::NONE;
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        for (i, (rows, dim)) in [(5usize, 64usize), (3, 7), (4, 33), (9, 1024), (6, 1536)].into_iter().enumerate() {
            let seed = 100 + i as u64;
            // The plain weighted norm.
            norm_case(rt, &NormCase { rows, dim, weight: true, pre: [none; 2], silu: false, post: none, scale: 1.0, resid: false }, 1e-6, seed);
            // The weightless embedder norm.
            norm_case(rt, &NormCase { rows, dim, weight: false, pre: [none; 2], silu: false, post: none, scale: 1.0, resid: false }, 1e-6, seed);
            // Both pre clamps, the activation, the post clamp (conv_norm).
            norm_case(rt, &NormCase { rows, dim, weight: true, pre: [c(-1.5, 2.0), c(-1.0, 1.2)], silu: true, post: c(-0.5, 0.9), scale: 1.0, resid: false }, 1e-5, seed);
            // The feed-forward's post norm: scale, then the residual.
            norm_case(rt, &NormCase { rows, dim, weight: true, pre: [c(-2.0, 2.5), none], silu: false, post: none, scale: 0.5, resid: true }, 1e-6, seed);
            // A lo > hi clamp gives hi (torch.clamp).
            norm_case(rt, &NormCase { rows, dim, weight: true, pre: [c(1.0, -1.0), none], silu: false, post: none, scale: 1.0, resid: false }, 1e-6, seed);
        }
    });
}

/// `out == resid == x` is the layer's own use (`rms_norm(y, x)` adds into the
/// residual in place); it must equal the out-of-place result bit for bit.
#[test]
fn rms_norm_adds_into_its_residual_in_place() {
    with_gpu(|rt| {
        let (rows, dim) = (7usize, 1024usize);
        let mut r = SplitMix::new(5);
        let y = rand(&mut r, rows * dim, 3.0, 0.0);
        let x = rand(&mut r, rows * dim, 2.0, 0.0);
        let w = buf(rt, &rand(&mut r, dim, 0.3, 1.0));
        let (yb, xb, out) = (buf(rt, &y), buf(rt, &x), seeded(rt, rows * dim, UNWRITTEN));
        let spec = |resid| RmsNorm {
            resid: Some(resid),
            scale: 0.5,
            pre: [Clamp::new(-2.0, 2.0).unwrap(), Clamp::NONE],
            ..RmsNorm::new(1e-6, Some(&w))
        };
        rms_norm(rt, &yb, &out, rows as u32, dim as u32, &spec(&xb)).unwrap();
        let want = read(rt, &out, rows * dim);
        let x2 = buf(rt, &x);
        rms_norm(rt, &yb, &x2, rows as u32, dim as u32, &spec(&x2)).unwrap();
        assert_eq!(read(rt, &x2, rows * dim), want);
    });
}

fn ew_case(rt: &Arc<GpuRuntime>, n: usize, pre: Clamp, silu_on: bool, post: Clamp, add: u8, cols: usize, seed: u64) {
    let mut r = SplitMix::new(seed);
    let x = rand(&mut r, n, 4.0, 0.0);
    let aux = rand(&mut r, if add == 2 { cols } else { n }, 2.0, 0.0);
    let (xb, ab) = (buf(rt, &x), buf(rt, &aux));
    let spec = Ew {
        pre,
        silu: silu_on,
        post,
        add: match add {
            0 => EwAdd::None,
            1 => EwAdd::Residual(&ab),
            _ => EwAdd::Bias(&ab, cols as u32),
        },
    };
    let out = seeded(rt, n, UNWRITTEN);
    ew(rt, &xb, &out, n as u32, &spec).unwrap();
    let want: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let a = clamp(f64::from(v), f64::from(pre.lo), f64::from(pre.hi));
            let a = if silu_on { silu(a) } else { a };
            let a = clamp(a, f64::from(post.lo), f64::from(post.hi));
            a + match add {
                0 => 0.0,
                1 => f64::from(aux[i]),
                _ => f64::from(aux[i % cols]),
            }
        })
        .collect();
    assert_close(&format!("ew n {n} silu {silu_on} add {add}"), &read(rt, &out, n), &want);
}

#[test]
fn ew_matches_the_f64_reference() {
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        for (n, cols) in [(37usize, 1usize), (64, 8), (7 * 4096, 4096), (13 * 1536, 1536)] {
            ew_case(rt, n, Clamp::NONE, false, Clamp::NONE, 0, cols, 1);
            // The feed-forward's middle: output clamp, SiLU, next input clamp.
            ew_case(rt, n, c(-2.5, 3.0), true, c(-1.0, 4.0), 0, cols, 2);
            // A bound inside SiLU's minimum (-0.278) and one past a huge input.
            ew_case(rt, n, c(-100.0, 100.0), true, c(-0.1, 0.1), 0, cols, 3);
            // The light conv's end: output clamp plus the residual.
            ew_case(rt, n, c(-1.0, 1.5), false, Clamp::NONE, 1, cols, 4);
            // The output projection's bias.
            ew_case(rt, n, Clamp::NONE, false, Clamp::NONE, 2, cols, 5);
        }
    });
}

#[test]
fn glu_matches_the_f64_reference() {
    with_gpu(|rt| {
        for (rows, hidden) in [(3usize, 5usize), (11, 64), (25, 1024)] {
            let mut r = SplitMix::new(rows as u64);
            let x = rand(&mut r, rows * 2 * hidden, 6.0, 0.0);
            let clamp_to = Clamp::new(-3.0, 4.0).unwrap();
            let out = seeded(rt, rows * hidden, UNWRITTEN);
            glu(rt, &buf(rt, &x), &out, rows as u32, hidden as u32, clamp_to).unwrap();
            let want: Vec<f64> = (0..rows * hidden)
                .map(|i| {
                    let (rw, c) = (i / hidden, i % hidden);
                    let a = clamp(f64::from(x[rw * 2 * hidden + c]), -3.0, 4.0);
                    let b = clamp(f64::from(x[rw * 2 * hidden + hidden + c]), -3.0, 4.0);
                    a * sigmoid(b)
                })
                .collect();
            assert_close(&format!("glu {rows}x{hidden}"), &read(rt, &out, rows * hidden), &want);
        }
    });
}

#[test]
fn dwconv1d_causal_matches_the_f64_reference() {
    with_gpu(|rt| {
        // Clips shorter than the taps, a single row, and the checkpoint's.
        for (frames, hidden, taps) in [(vec![3usize, 12, 1, 7], 9usize, 5usize), (vec![99, 330, 4], 1024, 5), (vec![6, 2], 16, 1)] {
            let mut r = SplitMix::new(hidden as u64);
            let table = ClipTable::new(rt, &frames).unwrap();
            let x = rand(&mut r, table.rows * hidden, 2.0, 0.0);
            let w = rand(&mut r, hidden * taps, 0.7, 0.0);
            let n = table.rows * hidden;
            let out = seeded(rt, n, UNWRITTEN);
            dwconv1d_causal(rt, &buf(rt, &x), &buf(rt, &w), &out, &table, hidden as u32, taps as u32).unwrap();
            let mut want = vec![0.0; n];
            for c in &table.clips {
                for t in 0..c.rows {
                    for ch in 0..hidden {
                        let mut s = 0.0;
                        for j in 0..taps {
                            // F.pad(x, (taps - 1, 0)) then conv1d (correlation).
                            let src = t as i64 + j as i64 - (taps as i64 - 1);
                            if src >= 0 {
                                s += f64::from(w[ch * taps + j]) * f64::from(x[(c.row0 + src as usize) * hidden + ch]);
                            }
                        }
                        want[(c.row0 + t) * hidden + ch] = s;
                    }
                }
            }
            assert_close(&format!("dwconv {} rows hidden {hidden} taps {taps}", table.rows), &read(rt, &out, n), &want);
        }
    });
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct AttnCase {
    heads: usize,
    head_dim: usize,
    chunk: usize,
    left: usize,
    cap: f64,
}

fn attn_case(rt: &Arc<GpuRuntime>, a: &AttnCase, frames: &[usize], seed: u64) {
    let (nh, hd) = (a.heads, a.head_dim);
    let hidden = nh * hd;
    let past = a.left - 1;
    let ctx = a.chunk + past;
    let mut r = SplitMix::new(seed);
    let table = ClipTable::new(rt, frames).unwrap();
    // The tower's mask per clip (any pattern), from `masks` over the rows.
    let oms = masks(&table.clips.iter().map(|c| c.rows).collect::<Vec<_>>());
    let rows = table.rows;
    let q = rand(&mut r, rows * hidden, 2.0, 0.0);
    let k = rand(&mut r, rows * hidden, 2.0, 0.0);
    let v = rand(&mut r, rows * hidden, 2.0, 0.0);
    let rel_w = rand(&mut r, hidden * hidden, 1.5 / (hidden as f64).sqrt(), 0.0);
    let pds = rand(&mut r, hd, 0.6, 0.0);
    let rel = relative_keys(hidden, ctx, &f64s(&rel_w));
    let qm = q_mult(&f64s(&pds));
    let rel32: Vec<f32> = rel.iter().map(|&v| v as f32).collect();
    let qm32: Vec<f32> = qm.iter().map(|&v| v as f32).collect();
    // Finite clamps that bite on the q, k, v loads and on the output.
    let (cq, ck, cv, co) = (
        Clamp::new(-1.2, 1.5).unwrap(),
        Clamp::new(-1.4, 1.1).unwrap(),
        Clamp::new(-1.3, 1.6).unwrap(),
        Clamp::new(-0.4, 0.5).unwrap(),
    );
    let k_scale = ((1.0f64 + 1f64.exp()).ln() / 2f64.ln()) as f32;
    let spec = AttentionSpec {
        heads: nh as u32,
        head_dim: hd as u32,
        chunk: a.chunk as u32,
        past: past as u32,
        k_scale,
        cap: a.cap as f32,
        invalid: -1e9,
        q: cq,
        k: ck,
        v: cv,
        out: co,
    };
    let out = seeded(rt, rows * hidden, UNWRITTEN);
    attention(
        rt,
        &buf(rt, &q),
        &buf(rt, &k),
        &buf(rt, &v),
        &buf(rt, &rel32),
        &buf(rt, &qm32),
        &buf_u32(rt, &mask_u32(&oms)),
        &out,
        &table,
        &spec,
    )
    .unwrap();
    let got = read(rt, &out, rows * hidden);

    let cl = |v: &[f32], c: Clamp| -> Vec<f64> { v.iter().map(|&x| clamp(f64::from(x), f64::from(c.lo), f64::from(c.hi))).collect() };
    let aspec = AttnSpec {
        heads: nh,
        head_dim: hd,
        chunk: a.chunk,
        left: a.left,
        cap: a.cap,
        invalid: -1e9,
    };
    let (qc, kc, vc) = (cl(&q, cq), cl(&k, ck), cl(&v, cv));
    // The reference sees the same f32-rounded relative keys and query scale
    // the kernel is given (they are host constants the model rounds once).
    let (rel_r, qm_r) = (f64s(&rel32), f64s(&qm32));
    let mut g = Vec::new();
    let mut e = Vec::new();
    for (i, c) in table.clips.iter().enumerate() {
        let span = c.row0 * hidden..(c.row0 + c.rows) * hidden;
        let want = attention_ref(&aspec, &qc[span.clone()], &kc[span.clone()], &vc[span.clone()], &rel_r, &qm_r, &oms[i]);
        g.extend_from_slice(&got[span]);
        e.extend(want.iter().map(|&v| clamp(v, f64::from(co.lo), f64::from(co.hi))));
    }
    assert_close(
        &format!("attention {nh} heads of {hd}, chunk {}, left {}, clips {frames:?}", a.chunk, a.left),
        &g,
        &e,
    );
}

#[test]
fn attention_matches_the_f64_reference() {
    with_gpu(|rt| {
        // Small, with a head dim that is not a multiple of the lane count.
        let small = AttnCase { heads: 3, head_dim: 10, chunk: 4, left: 5, cap: 3.0 };
        attn_case(rt, &small, &[14, 40, 3, 9, 100, 4, 1], 1);
        // Another window geometry: the past longer than the chunk.
        let long_past = AttnCase { heads: 2, head_dim: 33, chunk: 3, left: 8, cap: 5.0 };
        attn_case(rt, &long_past, &[10, 60, 13], 2);
        // The checkpoint's: 8 heads of 128, chunk 12, 12 frames of past,
        // clips of one, a few, a block, a block and a bit, and many blocks.
        let real = AttnCase { heads: 8, head_dim: 128, chunk: 12, left: 13, cap: 50.0 };
        attn_case(rt, &real, &[1, 11, 12, 13, 25, 83, 100, 4], 3);
        // Head dim 256, the most a lane holds.
        let wide = AttnCase { heads: 1, head_dim: 256, chunk: 6, left: 7, cap: 8.0 };
        attn_case(rt, &wide, &[30, 7], 4);
    });
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn the_wrappers_refuse_what_the_kernels_cannot_run() {
    with_gpu(|rt| {
        let err = |r: Result<(), String>| r.expect_err("accepted");
        // Clip tables.
        assert!(ClipTable::new(rt, &[]).is_err());
        assert!(ClipTable::new(rt, &[0]).is_err());
        assert!(ClipTable::new(rt, &[tessl::embedgemma2_audio::MAX_FRAMES_PER_FORWARD + 1]).is_err());
        assert!(ClipTable::new(rt, &vec![1; tessl::embedgemma2_audio::MAX_CLIPS + 1]).is_err());
        assert!(ClipTable::new(rt, &[tessl::embedgemma2_audio::MAX_FRAMES_PER_FORWARD, 1]).is_err());
        let table = ClipTable::new(rt, &[6, 4]).unwrap();
        let small = buf(rt, &[0.0; 4]);
        let ok = buf(rt, &vec![0.0; 100_000]);
        let mask = buf_u32(rt, &[1; 10]);
        // Conv: channel limits, eps, undersized buffers.
        assert!(err(sub_conv0_ln_relu(rt, &ok, &mask, &ok, &ok, &ok, &table, 8, 513, 1e-6)).contains("channels"));
        assert!(err(sub_conv0_ln_relu(rt, &ok, &mask, &ok, &ok, &ok, &table, 8, 4, 0.0)).contains("eps"));
        assert!(err(sub_conv0_ln_relu(rt, &small, &mask, &ok, &ok, &ok, &table, 8, 4, 1e-6)).contains("sub_conv0"));
        assert!(err(sub_conv1_ln_relu(rt, &ok, &mask, &ok, &ok, &ok, &table, 4, 4, 129, 1e-6)).contains("c1"));
        assert!(err(sub_conv1_ln_relu(rt, &ok, &mask, &small, &ok, &ok, &table, 4, 4, 4, 1e-6)).contains("sub_conv1"));
        // Norms and elementwise passes.
        assert!(err(rms_norm(rt, &small, &ok, 4, 4, &RmsNorm::new(1e-6, None))).contains("rms_norm"));
        assert!(err(rms_norm(rt, &ok, &ok, 4, 0, &RmsNorm::new(1e-6, None))).contains("dim"));
        assert!(err(rms_norm(rt, &ok, &ok, 4, 4, &RmsNorm::new(f32::NAN, None))).contains("eps"));
        assert!(err(rms_norm(rt, &ok, &ok, 4, 4, &RmsNorm { scale: f32::INFINITY, ..RmsNorm::new(1e-6, None) })).contains("scale"));
        assert!(err(ew(rt, &ok, &ok, 12, &Ew { pre: Clamp::NONE, silu: false, post: Clamp::NONE, add: EwAdd::Bias(&ok, 5) })).contains("bias"));
        assert!(err(ew(rt, &small, &ok, 12, &Ew { pre: Clamp::NONE, silu: false, post: Clamp::NONE, add: EwAdd::None })).contains("ew x"));
        assert!(Clamp::new(f32::NAN, 0.0).is_err());
        assert!(err(glu(rt, &small, &ok, 2, 4, Clamp::NONE)).contains("glu"));
        assert!(err(dwconv1d_causal(rt, &ok, &ok, &ok, &table, 4, 0)).contains("taps"));
        // Attention: the window the kernel is compiled for, head dim, cap.
        let spec = AttentionSpec {
            heads: 2,
            head_dim: 4,
            chunk: 4,
            past: 4,
            k_scale: 1.9,
            cap: 5.0,
            invalid: -1e9,
            q: Clamp::NONE,
            k: Clamp::NONE,
            v: Clamp::NONE,
            out: Clamp::NONE,
        };
        let call = |s: &AttentionSpec| attention(rt, &ok, &ok, &ok, &ok, &ok, &mask, &ok, &table, s);
        let rows_bad = AttentionSpec { head_dim: 257, ..spec };
        assert!(err(call(&rows_bad)).contains("unsupported"));
        assert!(err(call(&AttentionSpec { chunk: 60, ..spec })).contains("unsupported"));
        assert!(err(call(&AttentionSpec { cap: 0.0, ..spec })).contains("unsupported"));
        assert!(err(call(&AttentionSpec { past: 0, ..spec })).contains("unsupported"));
        // A window whose shifted rows the table would reach.
        assert!(err(call(&AttentionSpec { chunk: 30, past: 2, ..spec })).contains("unsupported"));
        // Outputs aliasing inputs.
        assert!(attention(rt, &ok, &ok, &ok, &ok, &ok, &mask, &ok, &table, &spec).is_err());
    });
}
