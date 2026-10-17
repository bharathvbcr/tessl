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
    attention, attention_store, ew, ew_store, lconv, lconv_store, rms_norm, rms_norm_store, rms_norm_then,
    rms_norm_then_store, sub_conv0_ln_relu, sub_conv1_ln_relu, ActStore, AttentionSpec, Clamp, ClipTable, Ew, Lconv,
    NormThen, RmsNorm,
};
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits_gpu, GpuBuffer};
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
        let (want, (_, to, fo), _) = conv_layer(
            &f64s(&feats[i]),
            (1, c.frames, mels),
            &ms[i],
            &f64s(&w),
            &f64s(&gamma),
            f64::from(eps),
        );
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
        assert_close(
            &format!("conv0 mels {mels} channels {channels} clip {i} ({} frames)", c.frames),
            &g,
            &e,
        );
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
    let mid: Vec<f32> = rand(&mut r, table.mid_rows * mels1 * c0, 1.0, 0.0)
        .iter()
        .map(|v| v.abs())
        .collect();
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
        let (want, (_, t2, f2), _) = conv_layer(
            &x,
            (c0, c.mid_rows, mels1),
            &m1,
            &f64s(&w_ck),
            &f64s(&gamma),
            f64::from(eps),
        );
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
            want.push(if c.resid {
                y + f64::from(res[row * c.dim + d])
            } else {
                y
            });
        }
    }
    assert_close(
        &format!(
            "rms_norm {}x{} weight {} silu {} resid {}",
            c.rows, c.dim, c.weight, c.silu, c.resid
        ),
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
        for (i, (rows, dim)) in [(5usize, 64usize), (3, 7), (4, 33), (9, 1024), (6, 1536)]
            .into_iter()
            .enumerate()
        {
            let seed = 100 + i as u64;
            // The plain weighted norm.
            norm_case(
                rt,
                &NormCase {
                    rows,
                    dim,
                    weight: true,
                    pre: [none; 2],
                    silu: false,
                    post: none,
                    scale: 1.0,
                    resid: false,
                },
                1e-6,
                seed,
            );
            // The weightless embedder norm.
            norm_case(
                rt,
                &NormCase {
                    rows,
                    dim,
                    weight: false,
                    pre: [none; 2],
                    silu: false,
                    post: none,
                    scale: 1.0,
                    resid: false,
                },
                1e-6,
                seed,
            );
            // Both pre clamps, the activation, the post clamp (conv_norm).
            norm_case(
                rt,
                &NormCase {
                    rows,
                    dim,
                    weight: true,
                    pre: [c(-1.5, 2.0), c(-1.0, 1.2)],
                    silu: true,
                    post: c(-0.5, 0.9),
                    scale: 1.0,
                    resid: false,
                },
                1e-5,
                seed,
            );
            // The feed-forward's post norm: scale, then the residual.
            norm_case(
                rt,
                &NormCase {
                    rows,
                    dim,
                    weight: true,
                    pre: [c(-2.0, 2.5), none],
                    silu: false,
                    post: none,
                    scale: 0.5,
                    resid: true,
                },
                1e-6,
                seed,
            );
            // A lo > hi clamp gives hi (torch.clamp).
            norm_case(
                rt,
                &NormCase {
                    rows,
                    dim,
                    weight: true,
                    pre: [c(1.0, -1.0), none],
                    silu: false,
                    post: none,
                    scale: 1.0,
                    resid: false,
                },
                1e-6,
                seed,
            );
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

fn ew_case(rt: &Arc<GpuRuntime>, n: usize, pre: Clamp, silu_on: bool, post: Clamp, seed: u64) {
    let mut r = SplitMix::new(seed);
    let x = rand(&mut r, n, 4.0, 0.0);
    let xb = buf(rt, &x);
    let spec = Ew {
        pre,
        silu: silu_on,
        post,
    };
    let out = seeded(rt, n, UNWRITTEN);
    ew(rt, &xb, &out, n as u32, &spec).unwrap();
    let want: Vec<f64> = x
        .iter()
        .map(|&v| {
            let a = clamp(f64::from(v), f64::from(pre.lo), f64::from(pre.hi));
            let a = if silu_on { silu(a) } else { a };
            clamp(a, f64::from(post.lo), f64::from(post.hi))
        })
        .collect();
    assert_close(&format!("ew n {n} silu {silu_on}"), &read(rt, &out, n), &want);
}

#[test]
fn ew_matches_the_f64_reference() {
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        for n in [37usize, 64, 7 * 4096, 13 * 1536] {
            ew_case(rt, n, Clamp::NONE, false, Clamp::NONE, 1);
            // The feed-forward's middle: output clamp, SiLU, next input clamp.
            ew_case(rt, n, c(-2.5, 3.0), true, c(-1.0, 4.0), 2);
            // A bound inside SiLU's minimum (-0.278) and one past a huge input.
            ew_case(rt, n, c(-100.0, 100.0), true, c(-0.1, 0.1), 3);
        }
    });
}

/// A stage without a norm (`RmsNorm::norm = false`) is what the light conv's
/// end and the output projection were as `ew`'s residual and bias modes: the
/// clamp plus `resid[row, d]`, the clamp plus `bias[d]`. Same sizes, seeds and
/// reference arithmetic as the `ew` cases they replace.
#[test]
fn rms_norm_without_a_norm_adds_a_residual_or_a_bias() {
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        for (rows, dim) in [(37usize, 1usize), (8, 8), (7, 4096), (13, 1536)] {
            let n = rows * dim;
            for (seed, bias, pre) in [(4u64, false, c(-1.0, 1.5)), (5, true, Clamp::NONE)] {
                let mut r = SplitMix::new(seed);
                let x = rand(&mut r, n, 4.0, 0.0);
                let aux = rand(&mut r, if bias { dim } else { n }, 2.0, 0.0);
                let (xb, ab) = (buf(rt, &x), buf(rt, &aux));
                let mut spec = RmsNorm::new(1e-6, None);
                spec.norm = false;
                spec.pre = [pre, Clamp::NONE];
                if bias {
                    spec.bias = Some(&ab);
                } else {
                    spec.resid = Some(&ab);
                }
                let out = seeded(rt, n, UNWRITTEN);
                rms_norm(rt, &xb, &out, rows as u32, dim as u32, &spec).unwrap();
                let want: Vec<f64> = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        clamp(f64::from(v), f64::from(pre.lo), f64::from(pre.hi))
                            + f64::from(if bias { aux[i % dim] } else { aux[i] })
                    })
                    .collect();
                assert_close(
                    &format!("no-norm stage {rows}x{dim} bias {bias}"),
                    &read(rt, &out, n),
                    &want,
                );
            }
        }
    });
}

/// `lconv` is the GLU, the depthwise causal conv and the conv norm in one pass.
/// The reference is each of the three written out in f64 (the GLU after the
/// clipped linear's output clamp, `F.pad(x, (taps - 1, 0))` then conv1d, the
/// weighted norm after the gradient clamp, SiLU, the next linear's input
/// clamp), over clips shorter than the taps, a single row, and the
/// checkpoint's shape. The shapes and seeds are those of the `glu` and
/// `dwconv1d_causal` cases this replaces.
#[allow(clippy::too_many_arguments)]
fn lconv_case(
    rt: &Arc<GpuRuntime>,
    frames: &[usize],
    hidden: usize,
    taps: usize,
    eps: f32,
    glu_c: Clamp,
    pre: Clamp,
    post: Clamp,
    seed: u64,
) {
    let mut r = SplitMix::new(seed);
    let table = ClipTable::new(rt, frames).unwrap();
    let x = rand(&mut r, table.rows * 2 * hidden, 6.0, 0.0);
    let w = rand(&mut r, hidden * taps, 0.7, 0.0);
    let nw = rand(&mut r, hidden, 0.3, 1.0);
    let n = table.rows * hidden;
    let out = seeded(rt, n, UNWRITTEN);
    let (xb, wb, nwb) = (buf(rt, &x), buf(rt, &w), buf(rt, &nw));
    lconv(
        rt,
        &xb,
        &wb,
        &out,
        &table,
        hidden as u32,
        taps as u32,
        &Lconv {
            eps,
            norm_weight: &nwb,
            glu: glu_c,
            pre,
            post,
        },
    )
    .unwrap();

    let g = |row: usize, ch: usize| {
        let at = |c: usize| {
            clamp(
                f64::from(x[row * 2 * hidden + c]),
                f64::from(glu_c.lo),
                f64::from(glu_c.hi),
            )
        };
        at(ch) * sigmoid(at(hidden + ch))
    };
    let mut want = vec![0.0; n];
    for c in &table.clips {
        for t in 0..c.rows {
            let conv: Vec<f64> = (0..hidden)
                .map(|ch| {
                    (0..taps)
                        .filter_map(|j| {
                            let src = t as i64 + j as i64 - (taps as i64 - 1);
                            (src >= 0).then(|| f64::from(w[ch * taps + j]) * g(c.row0 + src as usize, ch))
                        })
                        .sum()
                })
                .collect();
            let v: Vec<f64> = conv
                .iter()
                .map(|&y| clamp(y, f64::from(pre.lo), f64::from(pre.hi)))
                .collect();
            let normed = rms_eps(&v, Some(&f64s(&nw)), f64::from(eps));
            for (ch, y) in normed.into_iter().enumerate() {
                want[(c.row0 + t) * hidden + ch] = clamp(silu(y), f64::from(post.lo), f64::from(post.hi));
            }
        }
    }
    assert_close(
        &format!("lconv {} rows hidden {hidden} taps {taps}", table.rows),
        &read(rt, &out, n),
        &want,
    );
}

#[test]
fn lconv_matches_the_f64_reference() {
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        let (glu_c, pre, post) = (c(-3.0, 4.0), c(-1.5, 2.0), c(-0.5, 0.9));
        // Clips shorter than the taps, a single row, and the checkpoint's.
        lconv_case(rt, &[3, 12, 1, 7], 9, 5, 1e-6, glu_c, pre, post, 9);
        lconv_case(rt, &[99, 330, 4], 1024, 5, 1e-6, glu_c, pre, post, 1024);
        lconv_case(rt, &[6, 2], 16, 1, 1e-5, glu_c, pre, post, 16);
        // No clamp bites, then every one does; a width past a threadgroup's
        // threads (four channels per thread) and the widest the kernel holds.
        lconv_case(rt, &[11, 5], 64, 4, 1e-6, Clamp::NONE, Clamp::NONE, Clamp::NONE, 64);
        lconv_case(rt, &[9, 2], 64, 3, 1e-6, c(-0.3, 0.4), c(-0.2, 0.25), c(-0.1, 0.15), 65);
        lconv_case(rt, &[6, 3], 4096, 5, 1e-6, glu_c, pre, post, 4096);
    });
}

/// The first stage of the chained cases: clamped, scaled, and a residual, a
/// bias or neither added; with `no_norm` it has no norm and no weight.
fn first_stage<'a>(
    w: &'a GpuBuffer,
    resid: Option<&'a GpuBuffer>,
    bias: Option<&'a GpuBuffer>,
    no_norm: bool,
) -> RmsNorm<'a> {
    let mut s = RmsNorm::new(1e-6, (!no_norm).then_some(w));
    s.pre = [Clamp::new(-2.0, 2.5).unwrap(), Clamp::NONE];
    s.scale = 0.5;
    s.norm = !no_norm;
    s.resid = resid;
    s.bias = bias;
    s
}

/// A residual stage (clamped, scaled) and the weighted norm after it.
fn two_stages<'a>(w1: &'a GpuBuffer, w2: &'a GpuBuffer, resid: &'a GpuBuffer) -> (RmsNorm<'a>, RmsNorm<'a>) {
    let mut first = RmsNorm::new(1e-6, Some(w1));
    first.pre = [Clamp::new(-2.0, 2.5).unwrap(), Clamp::symmetric(9.0)];
    first.scale = 0.5;
    first.resid = Some(resid);
    (first, second_stage(w2))
}

/// A weighted norm after a `Clamp::symmetric(1.5)`.
fn second_stage(w: &GpuBuffer) -> RmsNorm<'_> {
    let mut s = RmsNorm::new(1e-6, Some(w));
    s.pre = [Clamp::symmetric(1.5), Clamp::NONE];
    s
}

/// The feed-forward's post-norm and residual add, as the layer writes them.
fn stream_post_norm<'a>(w: &'a GpuBuffer, resid: &'a GpuBuffer) -> RmsNorm<'a> {
    let mut s = RmsNorm::new(1e-6, Some(w));
    s.pre = [Clamp::new(-2.0, 2.5).unwrap(), Clamp::symmetric(9.0)];
    s.scale = 0.5;
    s.resid = Some(resid);
    s
}

/// `norm_out`.
fn stream_norm_out(w: &GpuBuffer) -> RmsNorm<'_> {
    let mut s = RmsNorm::new(1e-6, Some(w));
    s.pre = [Clamp::symmetric(9.0), Clamp::NONE];
    s
}

/// Two stages in one launch are the two kernels in sequence, bit for bit (the
/// reductions have the same shape), and each of a fan-out's outputs is the
/// second stage under its own clamp. Covers the layer's uses: a residual add
/// then a norm into three clamped copies, into one, the norm in place on the
/// residual stream, a no-norm stage then a norm, and a bias then a weightless
/// norm.
#[test]
fn rms_norm_then_is_two_rms_norms_in_one_launch() {
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        for (rows, dim, seed) in [
            (5usize, 64usize, 1u64),
            (3, 7, 2),
            (9, 1024, 3),
            (6, 1536, 4),
            (2, 4096, 5),
        ] {
            let n = rows * dim;
            let mut r = SplitMix::new(seed);
            let y = rand(&mut r, n, 3.0, 0.0);
            let x = rand(&mut r, n, 2.0, 0.0);
            let (w1, w2) = (rand(&mut r, dim, 0.3, 1.0), rand(&mut r, dim, 0.3, 1.0));
            let bias = rand(&mut r, dim, 1.0, 0.0);
            let (yb, w1b, w2b, biasb) = (buf(rt, &y), buf(rt, &w1), buf(rt, &w2), buf(rt, &bias));
            let posts = [c(-1.1, 1.3), c(-0.7, 0.9), c(-2.0, 0.4)];
            let (rows_u, dim_u) = (rows as u32, dim as u32);

            // The first stage's variants, each followed by a weighted norm.
            for (what, resid_in_place, bias_stage, no_norm) in [
                ("residual add then norm", true, false, false),
                ("no-norm residual then norm", true, false, true),
                ("bias then norm", false, true, true),
                ("plain norm then norm", false, false, false),
            ] {
                // Sequential: stage one into `mid`, stage two out of it.
                let xs = buf(rt, &x);
                let mid = seeded(rt, n, UNWRITTEN);
                let seq: Vec<GpuBuffer> = (0..3).map(|_| seeded(rt, n, UNWRITTEN)).collect();
                let first = first_stage(
                    &w1b,
                    resid_in_place.then_some(&xs),
                    bias_stage.then_some(&biasb),
                    no_norm,
                );
                rms_norm(rt, &yb, &mid, rows_u, dim_u, &first).unwrap();
                for (o, &p) in seq.iter().zip(&posts) {
                    let mut second = second_stage(&w2b);
                    second.post = p;
                    rms_norm(rt, &mid, o, rows_u, dim_u, &second).unwrap();
                }
                let want_mid = read(rt, &mid, n);
                let want: Vec<Vec<f64>> = seq.iter().map(|o| read(rt, o, n)).collect();

                // Chained, one to three outputs.
                for k in 1..=3usize {
                    let xs = buf(rt, &x);
                    let mid2 = seeded(rt, n, UNWRITTEN);
                    let outs: Vec<GpuBuffer> = (0..k).map(|_| seeded(rt, n, UNWRITTEN)).collect();
                    let then = NormThen {
                        spec: second_stage(&w2b),
                        outs: outs.iter().zip(&posts).map(|(o, &p)| (o, p)).collect(),
                    };
                    let first = first_stage(
                        &w1b,
                        resid_in_place.then_some(&xs),
                        bias_stage.then_some(&biasb),
                        no_norm,
                    );
                    rms_norm_then(rt, &yb, &mid2, rows_u, dim_u, &first, &then).unwrap();
                    assert_eq!(
                        read(rt, &mid2, n),
                        want_mid,
                        "{what} {rows}x{dim}: first stage's output"
                    );
                    for (i, o) in outs.iter().enumerate() {
                        assert_eq!(
                            read(rt, o, n),
                            want[i],
                            "{what} {rows}x{dim}: output {i} of {k} differs from the sequential launches"
                        );
                    }
                }
            }

            // The layer's in-place uses, against the out-of-place chain: the
            // residual add into the stream then `norm_out` in place on it.
            let xs = buf(rt, &x);
            let ys = buf(rt, &y);
            let stream = seeded(rt, n, UNWRITTEN);
            let xs2 = buf(rt, &x);
            rms_norm(rt, &ys, &stream, rows_u, dim_u, &stream_post_norm(&w1b, &xs2)).unwrap();
            let want_stream = {
                let o = seeded(rt, n, UNWRITTEN);
                rms_norm(rt, &stream, &o, rows_u, dim_u, &stream_norm_out(&w2b)).unwrap();
                read(rt, &o, n)
            };
            rms_norm_then(
                rt,
                &ys,
                &xs,
                rows_u,
                dim_u,
                &stream_post_norm(&w1b, &xs),
                &NormThen {
                    spec: stream_norm_out(&w2b),
                    outs: vec![(&xs, Clamp::NONE)],
                },
            )
            .unwrap();
            assert_eq!(
                read(rt, &xs, n),
                want_stream,
                "{rows}x{dim}: residual add then norm_out, in place"
            );
        }
    });
}

/// The chained launch against the f64 reference: the fan-out's three clamps
/// applied to one normed value, as the attention norm feeds q, k and v.
#[test]
fn rms_norm_then_matches_the_f64_reference() {
    with_gpu(|rt| {
        let (rows, dim) = (7usize, 1024usize);
        let n = rows * dim;
        let mut r = SplitMix::new(77);
        let y = rand(&mut r, n, 3.0, 0.0);
        let x = rand(&mut r, n, 2.0, 0.0);
        let (w1, w2) = (rand(&mut r, dim, 0.3, 1.0), rand(&mut r, dim, 0.3, 1.0));
        let (yb, xb, w1b, w2b) = (buf(rt, &y), buf(rt, &x), buf(rt, &w1), buf(rt, &w2));
        let mid = seeded(rt, n, UNWRITTEN);
        let posts = [
            Clamp::new(-1.1, 1.3).unwrap(),
            Clamp::new(-0.7, 0.9).unwrap(),
            Clamp::new(-2.0, 0.4).unwrap(),
        ];
        let outs: Vec<GpuBuffer> = (0..3).map(|_| seeded(rt, n, UNWRITTEN)).collect();
        let mut first = RmsNorm::new(1e-6, Some(&w1b));
        first.pre = [Clamp::new(-2.0, 2.5).unwrap(), Clamp::symmetric(9.0)];
        first.scale = 0.5;
        first.resid = Some(&xb);
        let mut second = RmsNorm::new(1e-6, Some(&w2b));
        second.pre = [Clamp::symmetric(1.5), Clamp::NONE];
        rms_norm_then(
            rt,
            &yb,
            &mid,
            rows as u32,
            dim as u32,
            &first,
            &NormThen {
                spec: second,
                outs: outs.iter().zip(&posts).map(|(o, &p)| (o, p)).collect(),
            },
        )
        .unwrap();

        let (mut want_mid, mut want_out) = (Vec::new(), vec![Vec::new(), Vec::new(), Vec::new()]);
        for row in 0..rows {
            let span = row * dim..(row + 1) * dim;
            let v: Vec<f64> = y[span.clone()]
                .iter()
                .map(|&t| clamp(clamp(f64::from(t), -2.0, 2.5), -9.0, 9.0))
                .collect();
            let m: Vec<f64> = rms_eps(&v, Some(&f64s(&w1)), 1e-6)
                .into_iter()
                .zip(&x[span])
                .map(|(a, &res)| a * 0.5 + f64::from(res))
                .collect();
            let v2: Vec<f64> = m.iter().map(|&t| clamp(t, -1.5, 1.5)).collect();
            let normed = rms_eps(&v2, Some(&f64s(&w2)), 1e-6);
            for (k, p) in posts.iter().enumerate() {
                want_out[k].extend(normed.iter().map(|&t| clamp(t, f64::from(p.lo), f64::from(p.hi))));
            }
            want_mid.extend(m);
        }
        assert_close("rms_norm_then first stage", &read(rt, &mid, n), &want_mid);
        for (k, o) in outs.iter().enumerate() {
            assert_close(&format!("rms_norm_then output {k}"), &read(rt, o, n), &want_out[k]);
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
        stride: hidden as u32,
        q_off: 0,
        k_off: 0,
        v_off: 0,
    };
    let (rel_b, qm_b, om_b) = (buf(rt, &rel32), buf(rt, &qm32), buf_u32(rt, &mask_u32(&oms)));
    // q, k, v in three buffers of their own, and in one fused q | k | v
    // projection's output (`3 * hidden` per row) read at their columns.
    let out = seeded(rt, rows * hidden, UNWRITTEN);
    attention(
        rt,
        &buf(rt, &q),
        &buf(rt, &k),
        &buf(rt, &v),
        &rel_b,
        &qm_b,
        &om_b,
        &out,
        &table,
        &spec,
    )
    .unwrap();
    let got = read(rt, &out, rows * hidden);
    let mut packed = vec![0.0f32; rows * 3 * hidden];
    for row in 0..rows {
        let dst = &mut packed[row * 3 * hidden..(row + 1) * 3 * hidden];
        dst[..hidden].copy_from_slice(&q[row * hidden..(row + 1) * hidden]);
        dst[hidden..2 * hidden].copy_from_slice(&k[row * hidden..(row + 1) * hidden]);
        dst[2 * hidden..].copy_from_slice(&v[row * hidden..(row + 1) * hidden]);
    }
    let pb = buf(rt, &packed);
    let out_fused = seeded(rt, rows * hidden, UNWRITTEN);
    attention(
        rt,
        &pb,
        &pb,
        &pb,
        &rel_b,
        &qm_b,
        &om_b,
        &out_fused,
        &table,
        &AttentionSpec {
            stride: 3 * hidden as u32,
            q_off: 0,
            k_off: hidden as u32,
            v_off: 2 * hidden as u32,
            ..spec
        },
    )
    .unwrap();
    let got_fused = read(rt, &out_fused, rows * hidden);
    assert_eq!(got_fused, got, "fused q|k|v layout differs from three buffers");

    let cl = |v: &[f32], c: Clamp| -> Vec<f64> {
        v.iter()
            .map(|&x| clamp(f64::from(x), f64::from(c.lo), f64::from(c.hi)))
            .collect()
    };
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
        let want = attention_ref(
            &aspec,
            &qc[span.clone()],
            &kc[span.clone()],
            &vc[span.clone()],
            &rel_r,
            &qm_r,
            &oms[i],
        );
        g.extend_from_slice(&got[span]);
        e.extend(want.iter().map(|&v| clamp(v, f64::from(co.lo), f64::from(co.hi))));
    }
    assert_close(
        &format!(
            "attention {nh} heads of {hd}, chunk {}, left {}, clips {frames:?}",
            a.chunk, a.left
        ),
        &g,
        &e,
    );
}

#[test]
fn attention_matches_the_f64_reference() {
    with_gpu(|rt| {
        // Small, with a head dim that is not a multiple of the lane count.
        let small = AttnCase {
            heads: 3,
            head_dim: 10,
            chunk: 4,
            left: 5,
            cap: 3.0,
        };
        attn_case(rt, &small, &[14, 40, 3, 9, 100, 4, 1], 1);
        // Another window geometry: the past longer than the chunk.
        let long_past = AttnCase {
            heads: 2,
            head_dim: 33,
            chunk: 3,
            left: 8,
            cap: 5.0,
        };
        attn_case(rt, &long_past, &[10, 60, 13], 2);
        // The checkpoint's: 8 heads of 128, chunk 12, 12 frames of past,
        // clips of one, a few, a block, a block and a bit, and many blocks.
        let real = AttnCase {
            heads: 8,
            head_dim: 128,
            chunk: 12,
            left: 13,
            cap: 50.0,
        };
        attn_case(rt, &real, &[1, 11, 12, 13, 25, 83, 100, 4], 3);
        // Head dim 256, the most a lane holds.
        let wide = AttnCase {
            heads: 1,
            head_dim: 256,
            chunk: 6,
            left: 7,
            cap: 8.0,
        };
        attn_case(rt, &wide, &[30, 7], 4);
    });
}

// ---------------------------------------------------------------------------
// bf16 outputs (`ActStore::Bf16`)
// ---------------------------------------------------------------------------
//
// A pass that feeds a bf16 GEMM stores its result as bf16. The arithmetic is
// the f32 pass's, so the contract is exact, not a tolerance: every bf16 output
// element is the f32 pass's element rounded to nearest even the way the GPU's
// own cast rounds it, and an output left in f32 beside it is bit for bit what
// the f32 pass wrote.

/// A bf16 buffer of `n` elements holding the bits of `UNWRITTEN`.
fn bf16_seeded(rt: &Arc<GpuRuntime>, n: usize) -> GpuBuffer {
    let b = rt.alloc_buffer(n.max(1) * 2).expect("alloc");
    b.write_bf16_bits(&vec![f32_to_bf16_bits_gpu(UNWRITTEN); n.max(1)]);
    b
}

fn read_bf16(rt: &Arc<GpuRuntime>, b: &GpuBuffer, n: usize) -> Vec<u16> {
    rt.synchronize().unwrap();
    b.try_contents_u16().unwrap()[..n].to_vec()
}

fn read_f32_bits(rt: &Arc<GpuRuntime>, b: &GpuBuffer, n: usize) -> Vec<f32> {
    rt.synchronize().unwrap();
    b.read_f32()[..n].to_vec()
}

/// `got` (bf16 bits) is `want` (the f32 pass's output) rounded, element by element.
fn assert_rounded(what: &str, rt: &Arc<GpuRuntime>, got: &GpuBuffer, want: &GpuBuffer, n: usize) {
    let want = read_f32_bits(rt, want, n);
    let got = read_bf16(rt, got, n);
    let moved = want
        .iter()
        .filter(|&&v| bf16_bits_to_f32(f32_to_bf16_bits_gpu(v)) != v)
        .count();
    assert!(
        moved > n / 4,
        "{what}: test bug: only {moved} of {n} values change when rounded"
    );
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, f32_to_bf16_bits_gpu(w), "{what}: element {i} (f32 pass wrote {w})");
    }
}

#[test]
fn row_passes_store_their_outputs_as_bf16_rounded() {
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        for (rows, dim) in [(5usize, 37usize), (7, 1024), (3, 1536)] {
            let n = rows * dim;
            let mut r = SplitMix::new(31 + dim as u64);
            let (y, x) = (rand(&mut r, n, 3.0, 0.0), rand(&mut r, n, 2.0, 0.0));
            let (w1, w2) = (rand(&mut r, dim, 0.3, 1.0), rand(&mut r, dim, 0.3, 1.0));
            let (yb, xb, w1b, w2b) = (buf(rt, &y), buf(rt, &x), buf(rt, &w1), buf(rt, &w2));
            let (rows_u, dim_u) = (rows as u32, dim as u32);

            // A lone stage (the feed-forward's pre-norm into its first GEMM),
            // with a weight, a clamp, SiLU and the next linear's clamp.
            let mut lone = RmsNorm::new(1e-6, Some(&w1b));
            lone.pre = [c(-2.0, 2.5), Clamp::symmetric(9.0)];
            lone.silu = true;
            lone.post = c(-0.2, 1.4);
            let f = seeded(rt, n, UNWRITTEN);
            rms_norm(rt, &yb, &f, rows_u, dim_u, &lone).unwrap();
            let b = bf16_seeded(rt, n);
            rms_norm_store(rt, &yb, &b, rows_u, dim_u, &lone, ActStore::Bf16).unwrap();
            assert_rounded(&format!("lone stage {rows}x{dim}"), rt, &b, &f, n);
            // F32 through the same entry is the f32 pass, bit for bit.
            let f2 = seeded(rt, n, UNWRITTEN);
            rms_norm_store(rt, &yb, &f2, rows_u, dim_u, &lone, ActStore::F32).unwrap();
            assert_eq!(read_f32_bits(rt, &f2, n), read_f32_bits(rt, &f, n));

            // Two stages, three outputs: each output's type is its own, and the
            // first stage's output (read back as f32) is untouched.
            let posts = [c(-1.1, 1.3), c(-0.7, 0.9), c(-2.0, 0.4)];
            let stages = |resid| two_stages(&w1b, &w2b, resid);
            let (mid_f, outs_f) = (
                seeded(rt, n, UNWRITTEN),
                (0..3).map(|_| seeded(rt, n, UNWRITTEN)).collect::<Vec<_>>(),
            );
            let (first, second) = stages(&xb);
            rms_norm_then(
                rt,
                &yb,
                &mid_f,
                rows_u,
                dim_u,
                &first,
                &NormThen {
                    spec: second,
                    outs: outs_f.iter().zip(&posts).map(|(o, &p)| (o, p)).collect(),
                },
            )
            .unwrap();
            for stores in [
                [ActStore::Bf16, ActStore::Bf16, ActStore::Bf16],
                [ActStore::F32, ActStore::Bf16, ActStore::F32],
                [ActStore::Bf16, ActStore::F32, ActStore::Bf16],
            ] {
                let mid = seeded(rt, n, UNWRITTEN);
                let outs: Vec<GpuBuffer> = stores
                    .iter()
                    .map(|s| match s {
                        ActStore::F32 => seeded(rt, n, UNWRITTEN),
                        ActStore::Bf16 => bf16_seeded(rt, n),
                    })
                    .collect();
                let (first, second) = stages(&xb);
                rms_norm_then_store(
                    rt,
                    &yb,
                    &mid,
                    rows_u,
                    dim_u,
                    &first,
                    &NormThen {
                        spec: second,
                        outs: outs.iter().zip(&posts).map(|(o, &p)| (o, p)).collect(),
                    },
                    stores,
                )
                .unwrap();
                assert_eq!(
                    read_f32_bits(rt, &mid, n),
                    read_f32_bits(rt, &mid_f, n),
                    "{stores:?}: first stage"
                );
                for (k, s) in stores.iter().enumerate() {
                    let what = format!("{rows}x{dim} output {k} of {stores:?}");
                    match s {
                        ActStore::Bf16 => assert_rounded(&what, rt, &outs[k], &outs_f[k], n),
                        ActStore::F32 => assert_eq!(
                            read_f32_bits(rt, &outs[k], n),
                            read_f32_bits(rt, &outs_f[k], n),
                            "{what}"
                        ),
                    }
                }
            }
        }
    });
}

#[test]
fn bf16_outputs_are_never_in_place_and_the_first_stage_stays_f32() {
    with_gpu(|rt| {
        let (rows, dim) = (4usize, 64usize);
        let n = rows * dim;
        let mut r = SplitMix::new(8);
        let (xb, rb) = (buf(rt, &rand(&mut r, n, 2.0, 0.0)), buf(rt, &rand(&mut r, n, 2.0, 0.0)));
        let w = buf(rt, &rand(&mut r, dim, 0.3, 1.0));
        let (rows_u, dim_u) = (rows as u32, dim as u32);
        let err = |r: Result<(), String>| r.expect_err("accepted");
        let spec = RmsNorm::new(1e-6, Some(&w));
        // An f32 buffer is big enough to hold the bf16 elements, but two bf16
        // elements share a word: storing over the input would race.
        let e = err(rms_norm_store(rt, &xb, &xb, rows_u, dim_u, &spec, ActStore::Bf16));
        assert!(e.contains("overlaps"), "{e}");
        let with_resid = RmsNorm {
            resid: Some(&rb),
            ..RmsNorm::new(1e-6, Some(&w))
        };
        let e = err(rms_norm_store(rt, &xb, &rb, rows_u, dim_u, &with_resid, ActStore::Bf16));
        assert!(e.contains("overlaps"), "{e}");
        // The same aliasing is allowed for f32 (each thread reads then writes its element).
        rms_norm_store(rt, &xb, &rb, rows_u, dim_u, &with_resid, ActStore::F32).unwrap();
        // A bf16 buffer one element short.
        let short = bf16_seeded(rt, n - 1);
        let e = err(rms_norm_store(rt, &xb, &short, rows_u, dim_u, &spec, ActStore::Bf16));
        assert!(e.contains("rms_norm out"), "{e}");
        // A chained launch's first output is read by the second stage as f32.
        let (mid, o0) = (bf16_seeded(rt, n), bf16_seeded(rt, n));
        let then = NormThen {
            spec: second_stage(&w),
            outs: vec![(&o0, Clamp::NONE)],
        };
        let e = err(rms_norm_then(rt, &xb, &mid, rows_u, dim_u, &spec, &then));
        assert!(e.contains("rms_norm"), "an f32 entry with bf16 buffers: {e}");
        // A bf16 second-stage output that is one of the launch's own inputs.
        let mid = seeded(rt, n, UNWRITTEN);
        let then_x = NormThen {
            spec: second_stage(&w),
            outs: vec![(&xb, Clamp::NONE)],
        };
        let e = err(rms_norm_then_store(
            rt,
            &xb,
            &mid,
            rows_u,
            dim_u,
            &spec,
            &then_x,
            [ActStore::Bf16; 3],
        ));
        assert!(e.contains("overlaps"), "{e}");
        let then_mid = NormThen {
            spec: second_stage(&w),
            outs: vec![(&mid, Clamp::NONE)],
        };
        let e = err(rms_norm_then_store(
            rt,
            &xb,
            &mid,
            rows_u,
            dim_u,
            &spec,
            &then_mid,
            [ActStore::Bf16; 3],
        ));
        assert!(e.contains("overlaps"), "{e}");
    });
}

#[test]
fn ew_stores_its_output_as_bf16_rounded_and_never_in_place() {
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        for n in [37usize, 7 * 4096] {
            let mut r = SplitMix::new(3);
            let xb = buf(rt, &rand(&mut r, n, 4.0, 0.0));
            let spec = Ew {
                pre: c(-2.5, 3.0),
                silu: true,
                post: c(-1.0, 4.0),
            };
            let f = seeded(rt, n, UNWRITTEN);
            ew(rt, &xb, &f, n as u32, &spec).unwrap();
            let b = bf16_seeded(rt, n);
            ew_store(rt, &xb, &b, n as u32, &spec, ActStore::Bf16).unwrap();
            assert_rounded(&format!("ew n {n}"), rt, &b, &f, n);
        }
        let xb = buf(rt, &[1.0; 64]);
        let spec = Ew {
            pre: Clamp::NONE,
            silu: false,
            post: Clamp::NONE,
        };
        let e = ew_store(rt, &xb, &xb, 64, &spec, ActStore::Bf16).expect_err("in place");
        assert!(e.contains("overlaps"), "{e}");
        let e = ew_store(rt, &xb, &bf16_seeded(rt, 63), 64, &spec, ActStore::Bf16).expect_err("short");
        assert!(e.contains("ew out"), "{e}");
    });
}

#[test]
fn lconv_and_attention_store_their_outputs_as_bf16_rounded() {
    let c = |lo: f32, hi: f32| Clamp::new(lo, hi).unwrap();
    with_gpu(|rt| {
        // The light conv's middle.
        let (frames, hidden, taps) = ([9usize, 20, 3], 64usize, 5usize);
        let table = ClipTable::new(rt, &frames).unwrap();
        let mut r = SplitMix::new(12);
        let x = buf(rt, &rand(&mut r, table.rows * 2 * hidden, 6.0, 0.0));
        let w = buf(rt, &rand(&mut r, hidden * taps, 0.7, 0.0));
        let nw = buf(rt, &rand(&mut r, hidden, 0.3, 1.0));
        let n = table.rows * hidden;
        let spec = Lconv {
            eps: 1e-6,
            norm_weight: &nw,
            glu: c(-3.0, 4.0),
            pre: c(-1.5, 2.0),
            post: c(-0.5, 0.9),
        };
        let f = seeded(rt, n, UNWRITTEN);
        lconv(rt, &x, &w, &f, &table, hidden as u32, taps as u32, &spec).unwrap();
        let b = bf16_seeded(rt, n);
        lconv_store(
            rt,
            &x,
            &w,
            &b,
            &table,
            hidden as u32,
            taps as u32,
            &spec,
            ActStore::Bf16,
        )
        .unwrap();
        assert_rounded("lconv", rt, &b, &f, n);
        let e = lconv_store(
            rt,
            &x,
            &w,
            &x,
            &table,
            hidden as u32,
            taps as u32,
            &spec,
            ActStore::Bf16,
        )
        .expect_err("in place");
        assert!(e.contains("overlaps"), "{e}");

        // The attention, three heads of 10, q|k|v in one projection's output.
        let (nh, hd, chunk, left) = (3usize, 10usize, 4usize, 5usize);
        let hidden = nh * hd;
        let past = left - 1;
        let ctx = chunk + past;
        let frames = [14usize, 40, 3];
        let table = ClipTable::new(rt, &frames).unwrap();
        let oms = masks(&table.clips.iter().map(|c| c.rows).collect::<Vec<_>>());
        let rows = table.rows;
        let packed = rand(&mut r, rows * 3 * hidden, 2.0, 0.0);
        let rel_w = rand(&mut r, hidden * hidden, 1.5 / (hidden as f64).sqrt(), 0.0);
        let rel: Vec<f32> = relative_keys(hidden, ctx, &f64s(&rel_w))
            .iter()
            .map(|&v| v as f32)
            .collect();
        let qm: Vec<f32> = q_mult(&f64s(&rand(&mut r, hd, 0.6, 0.0)))
            .iter()
            .map(|&v| v as f32)
            .collect();
        let spec = AttentionSpec {
            heads: nh as u32,
            head_dim: hd as u32,
            chunk: chunk as u32,
            past: past as u32,
            k_scale: ((1.0f64 + 1f64.exp()).ln() / 2f64.ln()) as f32,
            cap: 3.0,
            invalid: -1e9,
            q: c(-1.2, 1.5),
            k: c(-1.4, 1.1),
            v: c(-1.3, 1.6),
            out: c(-0.4, 0.5),
            stride: 3 * hidden as u32,
            q_off: 0,
            k_off: hidden as u32,
            v_off: 2 * hidden as u32,
        };
        let (pb, rel_b, qm_b, om_b) = (
            buf(rt, &packed),
            buf(rt, &rel),
            buf(rt, &qm),
            buf_u32(rt, &mask_u32(&oms)),
        );
        let n = rows * hidden;
        let f = seeded(rt, n, UNWRITTEN);
        attention(rt, &pb, &pb, &pb, &rel_b, &qm_b, &om_b, &f, &table, &spec).unwrap();
        let b = bf16_seeded(rt, n);
        attention_store(
            rt,
            &pb,
            &pb,
            &pb,
            &rel_b,
            &qm_b,
            &om_b,
            &b,
            &table,
            &spec,
            ActStore::Bf16,
        )
        .unwrap();
        assert_rounded("attention", rt, &b, &f, n);
        let e = attention_store(
            rt,
            &pb,
            &pb,
            &pb,
            &rel_b,
            &qm_b,
            &om_b,
            &pb,
            &table,
            &spec,
            ActStore::Bf16,
        )
        .expect_err("in place");
        assert!(e.contains("overlaps"), "{e}");
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
        assert!(err(sub_conv1_ln_relu(
            rt, &ok, &mask, &ok, &ok, &ok, &table, 4, 4, 129, 1e-6
        ))
        .contains("c1"));
        assert!(err(sub_conv1_ln_relu(
            rt, &ok, &mask, &small, &ok, &ok, &table, 4, 4, 4, 1e-6
        ))
        .contains("sub_conv1"));
        // Norms and elementwise passes.
        assert!(err(rms_norm(rt, &small, &ok, 4, 4, &RmsNorm::new(1e-6, None))).contains("rms_norm"));
        assert!(err(rms_norm(rt, &ok, &ok, 4, 0, &RmsNorm::new(1e-6, None))).contains("dim"));
        assert!(err(rms_norm(rt, &ok, &ok, 4, 4, &RmsNorm::new(f32::NAN, None))).contains("eps"));
        assert!(err(rms_norm(
            rt,
            &ok,
            &ok,
            4,
            4,
            &RmsNorm {
                scale: f32::INFINITY,
                ..RmsNorm::new(1e-6, None)
            }
        ))
        .contains("scale"));
        // A bias that is not one row wide, a residual and a bias at once.
        let five = buf(rt, &[0.0; 5]);
        assert!(err(rms_norm(
            rt,
            &ok,
            &ok,
            3,
            8,
            &RmsNorm {
                bias: Some(&five),
                norm: false,
                ..RmsNorm::new(1e-6, None)
            }
        ))
        .contains("bias"));
        assert!(err(rms_norm(
            rt,
            &ok,
            &ok,
            3,
            4,
            &RmsNorm {
                bias: Some(&ok),
                resid: Some(&ok),
                ..RmsNorm::new(1e-6, None)
            }
        ))
        .contains("not both"));
        assert!(err(ew(
            rt,
            &small,
            &ok,
            12,
            &Ew {
                pre: Clamp::NONE,
                silu: false,
                post: Clamp::NONE,
            }
        ))
        .contains("ew x"));
        assert!(Clamp::new(f32::NAN, 0.0).is_err());
        // The chained launch: output counts, a second stage that carries what
        // belongs to the first, outputs that alias what they must not.
        let first = RmsNorm::new(1e-6, None);
        let chain = |outs: Vec<(&GpuBuffer, Clamp)>| {
            rms_norm_then(
                rt,
                &ok,
                &ok,
                4,
                4,
                &first,
                &NormThen {
                    spec: RmsNorm::new(1e-6, None),
                    outs,
                },
            )
        };
        assert!(err(chain(vec![])).contains("outputs"));
        let more: Vec<GpuBuffer> = (0..4).map(|_| buf(rt, &vec![0.0; 100])).collect();
        assert!(err(chain(more.iter().map(|b| (b, Clamp::NONE)).collect())).contains("outputs"));
        assert!(err(rms_norm_then(
            rt,
            &ok,
            &ok,
            4,
            4,
            &first,
            &NormThen {
                spec: RmsNorm {
                    scale: 2.0,
                    ..RmsNorm::new(1e-6, None)
                },
                outs: vec![(&more[0], Clamp::NONE)],
            }
        ))
        .contains("second stage"));
        // An extra output that is the first stage's output, or another output.
        assert!(chain(vec![(&more[0], Clamp::NONE), (&ok, Clamp::NONE)]).is_err());
        assert!(chain(vec![
            (&more[0], Clamp::NONE),
            (&more[1], Clamp::NONE),
            (&more[1], Clamp::NONE)
        ])
        .is_err());
        // Light conv: hidden and taps limits, undersized buffers.
        let norm_w = buf(rt, &[1.0; 4]);
        let lc = |hidden: u32, taps: u32, x: &GpuBuffer| {
            lconv(
                rt,
                x,
                &ok,
                &more[0],
                &table,
                hidden,
                taps,
                &Lconv {
                    eps: 1e-6,
                    norm_weight: &norm_w,
                    glu: Clamp::NONE,
                    pre: Clamp::NONE,
                    post: Clamp::NONE,
                },
            )
        };
        assert!(err(lc(4, 0, &ok)).contains("taps"));
        assert!(err(lc(4097, 5, &ok)).contains("hidden"));
        assert!(err(lc(4, 5, &small)).contains("lconv"));
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
            stride: 8,
            q_off: 0,
            k_off: 0,
            v_off: 0,
        };
        let call = |s: &AttentionSpec| attention(rt, &ok, &ok, &ok, &ok, &ok, &mask, &ok, &table, s);
        // More past frames than the kernel carries partial logits for; a
        // masked logit too near the unmasked ones for a weight of exactly 0;
        // a column that does not fit the row stride.
        assert!(err(call(&AttentionSpec {
            past: 17,
            chunk: 1,
            ..spec
        }))
        .contains("unsupported"));
        assert!(err(call(&AttentionSpec {
            invalid: -100.0,
            ..spec
        }))
        .contains("unsupported"));
        assert!(err(call(&AttentionSpec { k_off: 1, ..spec })).contains("row stride"));
        let rows_bad = AttentionSpec { head_dim: 257, ..spec };
        assert!(err(call(&rows_bad)).contains("unsupported"));
        assert!(err(call(&AttentionSpec { chunk: 60, ..spec })).contains("unsupported"));
        assert!(err(call(&AttentionSpec { cap: 0.0, ..spec })).contains("unsupported"));
        assert!(err(call(&AttentionSpec { past: 0, ..spec })).contains("unsupported"));
        // A window whose shifted rows the table would reach.
        assert!(err(call(&AttentionSpec {
            chunk: 30,
            past: 2,
            ..spec
        }))
        .contains("unsupported"));
        // Outputs aliasing inputs.
        assert!(attention(rt, &ok, &ok, &ok, &ok, &ok, &mask, &ok, &table, &spec).is_err());
    });
}
