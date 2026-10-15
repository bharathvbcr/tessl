//! The kernels `tessl::embedgemma2_vision` adds for EmbeddingGemma 2's vision
//! tower, against f64 references, at a small fixture shape and at the model's
//! real ones (2376 patches in a 66 x 36 grid and 144 padding patches, 12 heads
//! of 64, a 10240-row position table, 768 columns), then adversarially.
//!
//! * `axial_qkv_norm_rope`: `vision_qkv_norm_rope_f32`, the `* w` q/k norms,
//!   the two-axis RoPE (x rotates the first half of a head, y the second),
//!   the weightless V norm and the K/V store.
//! * `patch_position_embed_add`: `vision_pos_embed_add_f32`, `proj + (x_emb +
//!   y_emb)`, and exactly `+0` on padding.
//! * `pool_by_members` with `plan_pooling`: `vision_pool_members_f32`, the
//!   pooler's cell averages, against the dense one-hot formulation
//!   transformers writes (`common::embedgemma2_vision::pool_dense`).
//!
//! Bounds, fixed before any run: an f32 kernel against an f64 reference within
//! `2e-5 * max|ref|` (one f32 rounding per step), the RoPE additionally within
//! `max_position * max_p |inv_freq_f32 - inv_freq_f64|` of angle (the
//! frequency table is rounded to f32 once, as torch's is, and the position
//! multiplies that rounding); the host pooling plan equal to the dense
//! reference's cells and sums.

mod common;

use std::sync::Arc;

use common::embedgemma2::rms;
use common::embedgemma2_vision::{pool_dense, rope_axial};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2_vision::{
    axial_qkv_norm_rope, patch_position_embed_add, plan_pooling, pool_by_members, AxialQkvShape, PoolPlan,
};
use tessl::nn::rope_inv_freq;
use tessl::qwen35::{AttnTargets, Cols};
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16, GpuBuffer};
use tessl::GpuRuntime;

const KERNEL_REL: f64 = 2e-5;

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

fn buf_i32(rt: &Arc<GpuRuntime>, data: &[i32]) -> GpuBuffer {
    buf_u32(rt, &data.iter().map(|&v| v as u32).collect::<Vec<_>>())
}

fn random(n: usize, seed: u64, scale: f64) -> Vec<f32> {
    let mut r = SplitMix::new(seed);
    (0..n).map(|_| (f64::from(r.unit()) * scale) as f32).collect()
}

/// `w x h` patches in raster order then `pad` padding patches.
fn grid(w: i32, h: i32, pad: usize) -> Vec<i32> {
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            v.extend([x, y]);
        }
    }
    v.extend(std::iter::repeat_n(-1, 2 * pad));
    v
}

fn max_ref(want: &[f64]) -> f64 {
    want.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

#[track_caller]
fn assert_close(what: &str, got: &[f32], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = max_ref(want).max(1e-30);
    let (mut worst, mut at) = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let e = (f64::from(g) - w).abs();
        if !e.is_finite() || e > worst {
            worst = if e.is_finite() { e } else { f64::INFINITY };
            at = i;
        }
    }
    eprintln!(
        "{what}: max abs {worst:.3e} of scale {scale:.3e} ({:.2e} relative)",
        worst / scale
    );
    assert!(
        worst <= rel * scale,
        "{what}: max abs error {worst:.3e} at {at} exceeds {rel:.1e} * {scale:.3e} (got {}, want {})",
        got[at],
        want[at]
    );
}

// ---------------------------------------------------------------------------
// plan_pooling against the dense reference (no device)
// ---------------------------------------------------------------------------

/// The plan's output, summed in f64 and averaged, as the dense reference's.
fn plan_rows(plan: &PoolPlan, hidden: &[f64], k: usize, h: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for s in 0..plan.cells.len() {
        let (a, b) = (plan.offsets[s] as usize, plan.offsets[s + 1] as usize);
        for j in 0..h {
            out.push(
                plan.members[a..b]
                    .iter()
                    .map(|&m| hidden[m as usize * h + j])
                    .sum::<f64>()
                    / (k * k) as f64,
            );
        }
    }
    out
}

#[test]
fn plan_pooling_matches_the_dense_reference() {
    let h = 3;
    // (width, height, padding patches, k): odd widths collide cells, a lone
    // patch, no padding, padding only in cell 0's favour, k = 2, 3 and 4, and
    // the two real shapes' grids.
    let cases = [
        (8, 6, 16, 2),
        (5, 4, 28, 2),
        (1, 1, 3, 2),
        (2, 2, 0, 2),
        (6, 3, 18, 3),
        (7, 5, 1, 3),
        (9, 4, 0, 3),
        (4, 4, 0, 4),
        (13, 9, 11, 4),
        (48, 48, 216, 3),
        (66, 36, 144, 3),
    ];
    let (mut checked, mut raised) = (0, 0);
    for (w, hh, pad, k) in cases {
        let pos = grid(w, hh, pad);
        let total = pos.len() / 2;
        let valid = (w * hh) as usize;
        let hidden: Vec<f64> = (0..total * h)
            .map(|i| {
                if i / h < valid {
                    f64::from(SplitMix::new(i as u64).unit())
                } else {
                    0.0
                }
            })
            .collect();
        let dense = std::panic::catch_unwind(|| pool_dense(&hidden, &pos, k, h));
        let plan = plan_pooling(&pos, valid, k as u32);
        match (dense, plan) {
            (Ok((rows, cells)), Ok(plan)) => {
                assert_eq!(
                    plan.cells.iter().map(|&c| c as usize).collect::<Vec<_>>(),
                    cells,
                    "{w}x{hh}+{pad} k={k}: cells"
                );
                let got = plan_rows(&plan, &hidden, k, h);
                assert_eq!(got.len(), rows.len());
                for (g, r) in got.iter().zip(&rows) {
                    assert!((g - r).abs() <= 1e-12, "{w}x{hh}+{pad} k={k}: {g} vs {r}");
                }
                assert_eq!(plan.members.len(), valid, "every real patch is in exactly one cell");
                assert_eq!(*plan.offsets.last().unwrap() as usize, valid);
                checked += 1;
            }
            // Where transformers raises (P not divisible, or a cell past the
            // output), so does the plan; neither is a case here that should.
            (Err(_), Err(_)) => raised += 1,
            (d, p) => panic!(
                "{w}x{hh}+{pad} k={k}: dense ok = {}, plan ok = {}",
                d.is_ok(),
                p.is_ok()
            ),
        }
    }
    // 7x5 and 9x4 grids at k = 3 and 13x9 at k = 4 land cells past the output
    // length (their rows of cells wrap into more rows than P / k² allows).
    assert!(
        checked >= 8 && raised >= 2,
        "test bug: {checked} cases pooled, {raised} raised"
    );
}

// ---------------------------------------------------------------------------
// axial_qkv_norm_rope
// ---------------------------------------------------------------------------

struct RopeCase {
    batch: usize,
    seq: usize,
    q_heads: usize,
    kv_heads: usize,
    d: usize,
    /// Cache positions per image (>= seq): the K/V stride the attention uses.
    cap: usize,
    /// Real patches per image; the rest of its `seq` rows are padding.
    real: Vec<usize>,
    grid_w: i32,
}

fn rope_inputs(c: &RopeCase, seed: u64) -> (Vec<i32>, Vec<f32>) {
    let width = (c.q_heads + 2 * c.kv_heads) * c.d;
    let mut positions = Vec::new();
    let mut proj = Vec::new();
    for b in 0..c.batch {
        let n = c.real[b];
        for t in 0..c.seq {
            if t < n {
                positions.extend([t as i32 % c.grid_w, t as i32 / c.grid_w]);
            } else {
                positions.extend([-1, -1]);
            }
        }
        let live = random(n * width, seed + b as u64, 1.0);
        proj.extend(live);
        proj.extend(std::iter::repeat_n(0.0f32, (c.seq - n) * width)); // padded rows: exactly zero
    }
    (positions, proj)
}

fn run_rope(rt: &Arc<GpuRuntime>, c: &RopeCase, positions: &[i32], proj: &[f32], qw: &[f32], kw: &[f32], theta: f32) {
    let width = (c.q_heads + 2 * c.kv_heads) * c.d;
    let rows = c.batch * c.seq;
    let pb = buf_f32(rt, proj);
    let (qb, kb) = (buf_f32(rt, qw), buf_f32(rt, kw));
    let pos = buf_i32(rt, positions);
    // Poison the outputs: every element the contract covers must be written.
    let nan = |n: usize| buf_f32(rt, &vec![f32::NAN; n]);
    let q_out = nan(rows * c.q_heads * c.d);
    let k_cache = nan(c.batch * c.cap * c.kv_heads * c.d);
    let v_cache = nan(c.batch * c.cap * c.kv_heads * c.d);
    axial_qkv_norm_rope(
        rt,
        &AxialQkvShape {
            batch: c.batch as u32,
            seq: c.seq as u32,
            q_heads: c.q_heads as u32,
            kv_heads: c.kv_heads as u32,
            head_dim: c.d as u32,
        },
        Cols::dense(&pb, width as u32),
        (c.q_heads * c.d) as u32,
        ((c.q_heads + c.kv_heads) * c.d) as u32,
        &qb,
        &kb,
        &AttnTargets {
            q_out: &q_out,
            k_cache: &k_cache,
            v_cache: &v_cache,
        },
        &pos,
        theta,
        1e-6,
    )
    .unwrap();
    rt.synchronize().unwrap();

    // The reference, row by row, in f64.
    let (d, kvh) = (c.d, c.kv_heads);
    let to64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let (qw64, kw64) = (to64(qw), to64(kw));
    let (mut want_q, mut want_k, mut want_v) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..rows {
        let row = to64(&proj[r * width..(r + 1) * width]);
        let (px, py) = (f64::from(positions[2 * r]), f64::from(positions[2 * r + 1]));
        for j in 0..c.q_heads {
            want_q.extend(rope_axial(
                &rms(&row[j * d..(j + 1) * d], Some(&qw64)),
                px,
                py,
                f64::from(theta),
            ));
        }
        for h in 0..kvh {
            let ko = (c.q_heads + h) * d;
            want_k.extend(rope_axial(
                &rms(&row[ko..ko + d], Some(&kw64)),
                px,
                py,
                f64::from(theta),
            ));
            let vo = (c.q_heads + kvh + h) * d;
            want_v.extend(rms(&row[vo..vo + d], None));
        }
    }
    let max_pos = positions.iter().copied().max().unwrap().max(1) as f64;
    let pairs = (d / 4) as u32;
    let gap = rope_inv_freq(pairs, d as u32 / 2, theta)
        .iter()
        .enumerate()
        .map(|(p, &f)| (f64::from(f) - 1.0 / f64::from(theta).powf(2.0 * p as f64 / (d / 2) as f64)).abs())
        .fold(0.0, f64::max);
    let rel = KERNEL_REL + max_pos * gap;
    let tag = format!("b{} t{} h{}/{} d{}", c.batch, c.seq, c.q_heads, kvh, d);
    assert_close(
        &format!("{tag} q"),
        &q_out.read_f32()[..rows * c.q_heads * d],
        &want_q,
        rel,
    );
    // K/V live in [batch, cap, kv_heads, d]: slot t of image b.
    let slots = |buf: &GpuBuffer| -> Vec<f32> {
        let all = buf.read_f32();
        let mut out = Vec::new();
        for b in 0..c.batch {
            for t in 0..c.seq {
                let o = (b * c.cap + t) * kvh * d;
                out.extend_from_slice(&all[o..o + kvh * d]);
            }
        }
        out
    };
    assert_close(&format!("{tag} k"), &slots(&k_cache), &want_k, rel);
    assert_close(&format!("{tag} v"), &slots(&v_cache), &want_v, KERNEL_REL);
    // Padded rows were zero in, so zero out (the weightless V norm of a zero
    // row is 0 * rsqrt(eps); the weighted ones likewise, rotated).
    for b in 0..c.batch {
        for t in c.real[b]..c.seq {
            let r = b * c.seq + t;
            assert!(
                q_out.read_f32()[r * c.q_heads * d..(r + 1) * c.q_heads * d]
                    .iter()
                    .all(|&v| v == 0.0),
                "{tag}: padded query row {r} is not zero"
            );
            let o = (b * c.cap + t) * kvh * d;
            assert!(
                k_cache.read_f32()[o..o + kvh * d].iter().all(|&v| v == 0.0),
                "{tag}: padded key {r}"
            );
            assert!(
                v_cache.read_f32()[o..o + kvh * d].iter().all(|&v| v == 0.0),
                "{tag}: padded value {r}"
            );
        }
    }
    // The cache slots past `seq` are not the kernel's: still the poison.
    if c.cap > c.seq {
        let all = k_cache.read_f32();
        assert!(
            all[c.seq * kvh * d..c.cap * kvh * d].iter().all(|v| v.is_nan()),
            "{tag}: wrote past seq"
        );
    }
}

fn norm_weights(d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let qk = (d as f64).powf(-0.25);
    let w = |s| {
        random(d, s, 0.3 * qk)
            .iter()
            .map(|&v| v + qk as f32)
            .collect::<Vec<f32>>()
    };
    (w(seed), w(seed + 1))
}

#[test]
fn axial_rope_matches_the_reference_at_the_fixture_shapes() {
    with_gpu(|rt| {
        // Two images, K/V capacity past seq, padded tails, MHA and GQA.
        for (q_heads, kv_heads, d) in [(3, 3, 32), (4, 2, 32), (2, 2, 64), (1, 1, 4), (2, 1, 8)] {
            let c = RopeCase {
                batch: 2,
                seq: 21,
                q_heads,
                kv_heads,
                d,
                cap: 27,
                real: vec![21, 13],
                grid_w: 5,
            };
            let (positions, proj) = rope_inputs(&c, 100 + d as u64);
            let (qw, kw) = norm_weights(d, 7);
            run_rope(rt, &c, &positions, &proj, &qw, &kw, 100.0);
        }
    });
}

#[test]
fn axial_rope_matches_the_reference_at_the_real_shape() {
    with_gpu(|rt| {
        // One image of 66 x 36 patches and 144 padding (2520 rows, 12 heads
        // of 64), alone; then two images with the second shorter, capacity
        // past seq.
        let one = RopeCase {
            batch: 1,
            seq: 2520,
            q_heads: 12,
            kv_heads: 12,
            d: 64,
            cap: 2520,
            real: vec![2376],
            grid_w: 66,
        };
        let (qw, kw) = norm_weights(64, 9);
        let (positions, proj) = rope_inputs(&one, 5);
        run_rope(rt, &one, &positions, &proj, &qw, &kw, 100.0);
        let two = RopeCase {
            batch: 2,
            seq: 1500,
            q_heads: 12,
            kv_heads: 12,
            d: 64,
            cap: 1600,
            real: vec![1500, 900],
            grid_w: 66,
        };
        let (positions, proj) = rope_inputs(&two, 6);
        run_rope(rt, &two, &positions, &proj, &qw, &kw, 100.0);
    });
}

/// The axes are not interchangeable and the halves are: rotating by (x, y) is
/// not rotating by (y, x), and a head's first half depends on x alone.
#[test]
fn axial_rope_rotates_the_first_half_by_x_and_the_second_by_y() {
    with_gpu(|rt| {
        let c = RopeCase {
            batch: 1,
            seq: 4,
            q_heads: 1,
            kv_heads: 1,
            d: 32,
            cap: 4,
            real: vec![4],
            grid_w: 2,
        };
        let (_, proj) = rope_inputs(&c, 3);
        let (qw, kw) = norm_weights(32, 1);
        let width = 3 * 32;
        let run = |pos: &[i32]| -> Vec<f32> {
            let (pb, qb, kb, pos) = (buf_f32(rt, &proj), buf_f32(rt, &qw), buf_f32(rt, &kw), buf_i32(rt, pos));
            let (q, k, v) = (
                buf_f32(rt, &[0.0; 4 * 32]),
                buf_f32(rt, &[0.0; 4 * 32]),
                buf_f32(rt, &[0.0; 4 * 32]),
            );
            axial_qkv_norm_rope(
                rt,
                &AxialQkvShape {
                    batch: 1,
                    seq: 4,
                    q_heads: 1,
                    kv_heads: 1,
                    head_dim: 32,
                },
                Cols::dense(&pb, width),
                32,
                64,
                &qb,
                &kb,
                &AttnTargets {
                    q_out: &q,
                    k_cache: &k,
                    v_cache: &v,
                },
                &pos,
                100.0,
                1e-6,
            )
            .unwrap();
            rt.synchronize().unwrap();
            q.read_f32()
        };
        // Row 1 of four, at (1, 2); the other rows stay put.
        let base = run(&[0, 0, 1, 2, 3, 1, 2, 2]);
        let moved_y = run(&[0, 0, 1, 5, 3, 1, 2, 2]);
        let moved_x = run(&[0, 0, 4, 2, 3, 1, 2, 2]);
        let row1 = 32..64; // q row 1 of the [4, 1, 32] output
        let (first, second) = (row1.start..row1.start + 16, row1.start + 16..row1.end);
        // y moves the second half and not the first; x the reverse.
        assert_eq!(base[first.clone()], moved_y[first.clone()], "the first half ignores y");
        assert_ne!(
            base[second.clone()],
            moved_y[second.clone()],
            "the second half follows y"
        );
        assert_eq!(
            base[second.clone()],
            moved_x[second.clone()],
            "the second half ignores x"
        );
        assert_ne!(base[first.clone()], moved_x[first], "the first half follows x");
        // The other rows do not depend on row 1's position.
        assert_eq!(base[..32], moved_y[..32]);
        assert_eq!(base[64..], moved_x[64..]);
        // Swapping the axes of (1, 2) is a different rotation.
        let swapped = run(&[0, 0, 2, 1, 3, 1, 2, 2]);
        assert_ne!(
            base[row1.clone()],
            swapped[row1],
            "(1, 2) and (2, 1) rotate differently"
        );
    });
}

#[test]
fn axial_rope_refuses_what_it_cannot_run() {
    with_gpu(|rt| {
        let (d, seq) = (32usize, 4usize);
        let width = 3 * d;
        let proj = buf_f32(rt, &vec![0.5; seq * width]);
        let w = buf_f32(rt, &vec![1.0; d]);
        let pos = buf_i32(rt, &vec![0; 2 * seq]);
        let out = |n: usize| buf_f32(rt, &vec![0.0; n]);
        let (q, k, v) = (out(seq * d), out(seq * d), out(seq * d));
        let shape = AxialQkvShape {
            batch: 1,
            seq: seq as u32,
            q_heads: 1,
            kv_heads: 1,
            head_dim: d as u32,
        };
        let call = |shape: &AxialQkvShape,
                    proj: Cols<'_>,
                    q_out: &GpuBuffer,
                    k_cache: &GpuBuffer,
                    pos: &GpuBuffer,
                    theta: f32| {
            axial_qkv_norm_rope(
                rt,
                shape,
                proj,
                d as u32,
                2 * d as u32,
                &w,
                &w,
                &AttnTargets {
                    q_out,
                    k_cache,
                    v_cache: &v,
                },
                pos,
                theta,
                1e-6,
            )
        };
        let dense = Cols::dense(&proj, width as u32);
        call(&shape, dense, &q, &k, &pos, 100.0).expect("the baseline call runs");
        // head_dim not a multiple of 4; heads not divisible; zero heads.
        assert!(
            call(&AxialQkvShape { head_dim: 30, ..shape }, dense, &q, &k, &pos, 100.0).is_err(),
            "head_dim 30"
        );
        assert!(
            call(&AxialQkvShape { q_heads: 3, ..shape }, dense, &q, &k, &pos, 100.0).is_err(),
            "3 q heads over 1 kv head"
        );
        assert!(
            call(&AxialQkvShape { kv_heads: 0, ..shape }, dense, &q, &k, &pos, 100.0).is_err(),
            "no kv heads"
        );
        // theta, buffers too small, aliasing, a window that does not fit.
        assert!(call(&shape, dense, &q, &k, &pos, 0.0).is_err(), "theta 0");
        assert!(call(&shape, dense, &q, &k, &pos, f32::NAN).is_err(), "theta NaN");
        assert!(
            call(&shape, dense, &out(seq * d - 1), &k, &pos, 100.0).is_err(),
            "short q_out"
        );
        assert!(
            call(&shape, dense, &q, &out(seq * d - 1), &pos, 100.0).is_err(),
            "short caches"
        );
        assert!(
            call(&shape, dense, &q, &k, &buf_i32(rt, &vec![0; 2 * seq - 1]), 100.0).is_err(),
            "short positions"
        );
        assert!(
            call(&shape, dense, &proj, &k, &pos, 100.0).is_err(),
            "q_out aliasing the projection"
        );
        assert!(
            call(&shape, dense, &q, &q, &pos, 100.0).is_err(),
            "k cache aliasing q_out"
        );
        assert!(
            call(
                &shape,
                Cols {
                    buf: &proj,
                    ld: width as u32,
                    off: 8
                },
                &q,
                &k,
                &pos,
                100.0
            )
            .is_err(),
            "a window past its row"
        );
        // Zero images or patches do nothing.
        call(&AxialQkvShape { batch: 0, ..shape }, dense, &q, &k, &pos, 100.0).unwrap();
        // Caches smaller than seq positions.
        let (small_k, small_v) = (out((seq - 1) * d), out((seq - 1) * d));
        assert!(
            axial_qkv_norm_rope(
                rt,
                &shape,
                dense,
                d as u32,
                2 * d as u32,
                &w,
                &w,
                &AttnTargets {
                    q_out: &q,
                    k_cache: &small_k,
                    v_cache: &small_v
                },
                &pos,
                100.0,
                1e-6
            )
            .is_err(),
            "caches shorter than seq"
        );
    });
}

// ---------------------------------------------------------------------------
// patch_position_embed_add
// ---------------------------------------------------------------------------

fn check_pos_embed(rt: &Arc<GpuRuntime>, table_len: usize, dim: usize, positions: &[i32], seed: u64) {
    let rows = positions.len() / 2;
    let table: Vec<f32> = random(2 * table_len * dim, seed, 0.5);
    let bits = f32_slice_to_bf16(&table);
    let t64: Vec<f64> = bits.iter().map(|&b| f64::from(bf16_bits_to_f32(b))).collect();
    let tb = rt.alloc_buffer(bits.len() * 2).unwrap();
    tb.write_bf16_bits(&bits);
    let proj = random(rows * dim, seed + 1, 1.0);
    let hb = buf_f32(rt, &proj);
    let pb = buf_i32(rt, positions);
    patch_position_embed_add(rt, &hb, &tb, &pb, rows as u32, dim as u32, table_len as u32).unwrap();
    rt.synchronize().unwrap();
    let got = hb.read_f32();
    let mut want = Vec::with_capacity(rows * dim);
    for r in 0..rows {
        let (x, y) = (positions[2 * r], positions[2 * r + 1]);
        for j in 0..dim {
            want.push(if x == -1 && y == -1 {
                0.0
            } else {
                f64::from(proj[r * dim + j]) + (t64[x as usize * dim + j] + t64[(table_len + y as usize) * dim + j])
            });
        }
    }
    assert_close(
        &format!("pos embed {rows}x{dim} table {table_len}"),
        &got[..rows * dim],
        &want,
        1e-6,
    );
    for r in 0..rows {
        if positions[2 * r] == -1 && positions[2 * r + 1] == -1 {
            assert!(
                got[r * dim..(r + 1) * dim].iter().all(|v| v.to_bits() == 0),
                "padding row {r} is not exactly +0"
            );
        }
    }
}

#[test]
fn position_embeddings_match_the_reference() {
    with_gpu(|rt| {
        // Fixture shape.
        check_pos_embed(rt, 16, 24, &grid(5, 4, 3), 1);
        check_pos_embed(rt, 16, 24, &grid(1, 1, 0), 2);
        // The real shapes: the 10240-row table, 768 columns, both images.
        check_pos_embed(rt, 10240, 768, &grid(48, 48, 216), 3);
        check_pos_embed(rt, 10240, 768, &grid(66, 36, 144), 4);
    });
}

#[test]
fn position_embeddings_stay_in_bounds_and_refuse_bad_buffers() {
    with_gpu(|rt| {
        let (len, dim, rows) = (8usize, 4usize, 3usize);
        let table: Vec<f32> = random(2 * len * dim, 1, 1.0);
        let bits = f32_slice_to_bf16(&table);
        let tb = rt.alloc_buffer(bits.len() * 2).unwrap();
        tb.write_bf16_bits(&bits);
        let t64: Vec<f64> = bits.iter().map(|&b| f64::from(bf16_bits_to_f32(b))).collect();
        // Device positions past the table (or one coordinate -1) clamp into it.
        let hb = buf_f32(rt, &vec![0.0; rows * dim]);
        let pos = buf_i32(rt, &[1_000_000, 3, 2, -1, i32::MIN, i32::MAX]);
        patch_position_embed_add(rt, &hb, &tb, &pos, rows as u32, dim as u32, len as u32).unwrap();
        rt.synchronize().unwrap();
        let got = hb.read_f32();
        let clamped = [(len - 1, 3), (2, 0), (0, len - 1)];
        for (r, (x, y)) in clamped.iter().enumerate() {
            for j in 0..dim {
                let want = t64[x * dim + j] + t64[(len + y) * dim + j];
                assert!((f64::from(got[r * dim + j]) - want).abs() <= 1e-6, "row {r} col {j}");
            }
        }
        let h2 = buf_f32(rt, &vec![0.0; rows * dim]);
        let p2 = buf_i32(rt, &vec![0; 2 * rows]);
        assert!(
            patch_position_embed_add(rt, &h2, &tb, &p2, rows as u32 + 1, dim as u32, len as u32).is_err(),
            "rows past h"
        );
        assert!(
            patch_position_embed_add(rt, &h2, &tb, &p2, rows as u32, dim as u32, len as u32 + 1).is_err(),
            "table too short"
        );
        assert!(
            patch_position_embed_add(rt, &h2, &tb, &p2, rows as u32, 0, len as u32).is_err(),
            "dim 0"
        );
        assert!(
            patch_position_embed_add(rt, &h2, &tb, &p2, rows as u32, dim as u32, 0).is_err(),
            "table_len 0"
        );
        assert!(
            patch_position_embed_add(rt, &h2, &tb, &h2, rows as u32, dim as u32, len as u32).is_err(),
            "h aliasing positions"
        );
        patch_position_embed_add(rt, &h2, &tb, &p2, 0, dim as u32, len as u32).unwrap();
    });
}

// ---------------------------------------------------------------------------
// pool_by_members
// ---------------------------------------------------------------------------

fn upload_plan(rt: &Arc<GpuRuntime>, plan: &PoolPlan) -> (GpuBuffer, GpuBuffer) {
    (buf_u32(rt, &plan.offsets), buf_u32(rt, &plan.members))
}

fn check_pool(rt: &Arc<GpuRuntime>, w: i32, h: i32, pad: usize, k: usize, dim: usize, seed: u64) {
    let pos = grid(w, h, pad);
    let total = pos.len() / 2;
    let valid = (w * h) as usize;
    let plan = plan_pooling(&pos, valid, k as u32).unwrap();
    // Real rows carry data; padding rows are zero, as the pooler masks them.
    let mut hidden = random(valid * dim, seed, 1.0);
    hidden.resize(total * dim, 0.0);
    let (ob, mb) = upload_plan(rt, &plan);
    let hb = buf_f32(rt, &hidden);
    let s = plan.cells.len();
    let out = buf_f32(rt, &vec![f32::NAN; s * dim]);
    let scale = (dim as f32).sqrt();
    pool_by_members(
        rt,
        &hb,
        &ob,
        &mb,
        &out,
        s as u32,
        total as u32,
        dim as u32,
        plan.members.len() as u32,
        1.0 / (k * k) as f32,
        scale,
    )
    .unwrap();
    rt.synchronize().unwrap();
    let h64: Vec<f64> = hidden.iter().map(|&v| f64::from(v)).collect();
    let (rows, cells) = pool_dense(&h64, &pos, k, dim);
    assert_eq!(plan.cells.iter().map(|&c| c as usize).collect::<Vec<_>>(), cells);
    let want: Vec<f64> = rows.iter().map(|v| v * f64::from(scale)).collect();
    let got = out.read_f32();
    assert_close(
        &format!("pool {w}x{h}+{pad} k={k} dim={dim}"),
        &got[..s * dim],
        &want,
        KERNEL_REL,
    );
    for (row, &cell) in plan.cells.iter().enumerate() {
        let empty = plan.offsets[row] == plan.offsets[row + 1];
        if empty {
            assert_eq!(cell, 0, "only cell 0 can be kept empty");
            assert!(
                got[row * dim..(row + 1) * dim].iter().all(|v| v.to_bits() == 0),
                "an empty cell is exactly +0"
            );
        }
    }
}

#[test]
fn pooling_matches_the_dense_reference() {
    with_gpu(|rt| {
        // Fixture shapes, including colliding cells and an empty cell 0.
        check_pool(rt, 8, 6, 16, 2, 5, 1);
        check_pool(rt, 5, 4, 28, 2, 5, 2);
        check_pool(rt, 6, 3, 18, 3, 7, 3);
        check_pool(rt, 1, 1, 3, 2, 3, 4);
        // The real shapes: 2520 rows of 768, 256 and 264 soft tokens.
        check_pool(rt, 48, 48, 216, 3, 768, 5);
        check_pool(rt, 66, 36, 144, 3, 768, 6);
    });
}

/// An empty cell 0: the one patch is in cell 1, and the padding keeps cell 0.
#[test]
fn an_empty_kept_cell_is_exactly_zero() {
    with_gpu(|rt| {
        let mut pos = vec![3, 0];
        pos.extend(std::iter::repeat_n(-1, 34));
        let plan = plan_pooling(&pos, 1, 3).unwrap();
        assert_eq!(plan.offsets, vec![0, 0, 1]);
        let (ob, mb) = upload_plan(rt, &plan);
        let hb = buf_f32(rt, &random(18 * 4, 1, 1.0));
        let out = buf_f32(rt, &[f32::NAN; 2 * 4]);
        pool_by_members(rt, &hb, &ob, &mb, &out, 2, 18, 4, 1, 1.0 / 9.0, 2.0).unwrap();
        rt.synchronize().unwrap();
        let got = out.read_f32();
        assert!(got[..4].iter().all(|v| v.to_bits() == 0), "{:?}", &got[..4]);
        assert!(got[4..8].iter().all(|v| v.is_finite() && *v != 0.0));
    });
}

/// The offsets and members are device values: a range past `n_members` clamps,
/// a reversed range is empty, and a member row at or past `rows` is skipped.
#[test]
fn pooling_survives_hostile_device_ranges() {
    with_gpu(|rt| {
        let (rows, dim) = (5usize, 3usize);
        let hidden: Vec<f32> = (0..rows * dim).map(|i| i as f32 + 1.0).collect();
        let hb = buf_f32(rt, &hidden);
        let offsets = buf_u32(rt, &[0, 2, 1, 4, 99, u32::MAX]);
        let members = buf_u32(rt, &[0, 4, 7, 1]);
        let out = buf_f32(rt, &vec![f32::NAN; 5 * dim]);
        pool_by_members(
            rt,
            &hb,
            &offsets,
            &members,
            &out,
            5,
            rows as u32,
            dim as u32,
            4,
            1.0,
            1.0,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let got = out.read_f32();
        let row = |r: usize| -> Vec<f32> { hidden[r * dim..(r + 1) * dim].to_vec() };
        let add = |a: Vec<f32>, b: Vec<f32>| -> Vec<f32> { a.iter().zip(&b).map(|(x, y)| x + y).collect() };
        // [0, 2): rows 0 and 4. [2, 1): reversed, empty. [1, 4): member 4's row 4, then 7 (skipped), then 1.
        assert_eq!(got[..3], add(row(0), row(4))[..]);
        assert!(got[3..6].iter().all(|v| v.to_bits() == 0), "a reversed range is +0");
        assert_eq!(got[6..9], add(row(4), row(1))[..]);
        // [4, 99): clamped to n_members = 4, so empty from 4; [99, MAX): empty.
        assert!(got[9..15].iter().all(|v| v.to_bits() == 0));
    });
}

#[test]
fn pooling_refuses_bad_buffers() {
    with_gpu(|rt| {
        let hb = buf_f32(rt, &[1.0; 6 * 2]);
        let offsets = buf_u32(rt, &[0, 1, 2]);
        let members = buf_u32(rt, &[0, 1]);
        let out = buf_f32(rt, &[0.0; 2 * 2]);
        let call = |h: &GpuBuffer, o: &GpuBuffer, m: &GpuBuffer, out: &GpuBuffer, s: u32, n: u32, w: f32| {
            pool_by_members(rt, h, o, m, out, s, 6, 2, n, w, 1.0)
        };
        call(&hb, &offsets, &members, &out, 2, 2, 1.0).expect("baseline");
        assert!(
            call(&hb, &offsets, &members, &out, 3, 2, 1.0).is_err(),
            "offsets for 2 cells, asked 3"
        );
        assert!(
            call(&hb, &offsets, &members, &out, 2, 3, 1.0).is_err(),
            "members for 2, asked 3"
        );
        assert!(
            call(&hb, &offsets, &members, &out, 2, 2, f32::NAN).is_err(),
            "weight NaN"
        );
        assert!(
            call(&hb, &offsets, &members, &buf_f32(rt, &[0.0; 3]), 2, 2, 1.0).is_err(),
            "short out"
        );
        assert!(call(&hb, &offsets, &members, &hb, 2, 2, 1.0).is_err(), "out aliasing h");
        assert!(
            pool_by_members(
                rt,
                &buf_f32(rt, &[0.0; 11]),
                &offsets,
                &members,
                &out,
                2,
                6,
                2,
                2,
                1.0,
                1.0
            )
            .is_err(),
            "h shorter than rows * dim"
        );
        call(&hb, &offsets, &members, &out, 0, 2, 1.0).expect("no output rows is a no-op");
    });
}
