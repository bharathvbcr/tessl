//! The bf16-tile prefill attention kernels (`qwen35_attn_tiled_bf16*_h256_*`,
//! `attn_prefill_bf16::attn_prefill_bf16_with_tile`) against an f64 oracle,
//! with an error bound derived from bf16 rounding rather than chosen.
//!
//! What the kernels do differently from the f32 `attn_prefill`: Q, K and V are
//! bf16 in device memory, so `Q.K^T` runs on bf16 operands, and (for the `bf16`
//! entries) the probability tile P is rounded to bf16 before `P.V`. Scores, the
//! online softmax and the accumulator stay f32. Two comparisons, two bounds, both
//! per output element:
//!
//! - Tier 2 (every case): the oracle takes the *bf16-rounded* Q, K and V, the
//!   operands the kernel reads, in f64. The kernel's error is then P's rounding,
//!   f32 accumulation and the fast-math `exp`.
//! - Tier 1 (the moderate-score cases): the oracle takes the *original f32*
//!   Q, K and V, so the error also holds the rounding of the inputs to bf16, the
//!   error a producer that writes bf16 adds.
//!
//! The bound. Let `u = 2^-8` be bf16's unit roundoff (7 stored mantissa bits,
//! round to nearest), `w_j` the exact softmax weights of a row, `A_x = sum_j
//! w_j |v_jx|`, and `s_j = scale * q.k_j`.
//!
//! - Rounding q and k to bf16 moves a dot product by at most
//!   `(2u + u^2) * sum_i |q_i k_ji|`, so every scaled score moves by at most
//!   `d_in = (2u + u^2) * scale * dq`, `dq = max_j sum_i |q_i k_ji|`: the
//!   dependence on `Q.K` is through the row's absolute dot product, not its
//!   signed value.
//! - Rounding P to bf16 moves each weight by a relative `u`. The f32 score
//!   accumulation adds `D * 2^-24 * scale * dq`, and Metal's fast `exp` is good
//!   to `3 + floor(2|x|)` ulps, `|x|` up to the row's score spread.
//!
//! Call the sum of those relative perturbations of the log-weights `a`. A weight
//! moves by a factor within `e^{+-a}` before normalization, so within `e^{+-2a}`
//! after it, and the normalized weights move by at most `w_j (e^{2a} - 1)`. The
//! output is `sum_j w_j v_jx`, so
//!
//! `|dO_x| <= ((e^{2a} - 1)(1 + u_v) + u_v + g) * A_x`
//!
//! with `u_v = u` when V's rounding is in the comparison (tier 1) and 0 when it
//! is not, and `g = (live keys + D) * 2^-24` for the f32 accumulations of P.V,
//! the online rescales and the final `1/l`.

mod common;

use common::{buf, buf_bf16, buf_u32, random_f32, round_trip_bf16, seeded, with_gpu, SplitMix};
use tessl::attn_prefill_bf16::{attn_prefill_bf16_entry, attn_prefill_bf16_with_tile, AttnProbTile};
use tessl::nn::AttnDims;
use tessl::qwen35::AttnTile;
use tessl::tensor::f32_to_bf16_bits_gpu;

const D: usize = 256;
const U: f64 = 1.0 / 256.0;
const F32_EPS: f64 = 1.0 / 16_777_216.0;
const UNWRITTEN: f32 = -6.5e28;
const METAL: &str = include_str!("../kernels/qwen35_attn_tiled.metal");

/// Every compiled (tile, probability tile) pair.
fn variants() -> Vec<(AttnTile, AttnProbTile)> {
    let mut v: Vec<_> = AttnTile::ALL.iter().map(|&t| (t, AttnProbTile::Bf16)).collect();
    v.push((AttnTile::Q32K32Sg4, AttnProbTile::F32));
    v
}

/// One attention call: `[batch, tq, h, D]` queries at positions `q_off..`, a
/// `[batch, cap, hkv, D]` cache of which `tkv` positions are live, causal from
/// position 0.
struct Case {
    label: String,
    batch: usize,
    tq: usize,
    q_off: usize,
    tkv: usize,
    cap: usize,
    h: usize,
    hkv: usize,
    scale: f32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// Query positions (within `tq`) the oracle checks; all of them for small cases.
    rows: Vec<usize>,
    /// Whether the input-rounding comparison (tier 1) applies.
    tier1: bool,
}

impl Case {
    fn q_len(&self) -> usize {
        self.batch * self.tq * self.h * D
    }
    /// Overwrite the cache positions the call does not read with a value that
    /// would dominate any softmax, so a mask or length bug cannot hide.
    fn poison_unread_kv(&mut self, from: usize) {
        for b in 0..self.batch {
            for j in from..self.cap {
                for hk in 0..self.hkv {
                    let at = ((b * self.cap + j) * self.hkv + hk) * D;
                    self.k[at..at + D].fill(3.0e4);
                    self.v[at..at + D].fill(3.0e4);
                }
            }
        }
    }
}

/// A random case: Q and K uniform in `[-s_qk, s_qk)`, V in `[-1, 1)`.
#[allow(clippy::too_many_arguments)]
fn random_case(
    label: &str,
    batch: usize,
    tq: usize,
    q_off: usize,
    tkv: usize,
    cap: usize,
    scale: f32,
    s_qk: f32,
    seed: u64,
) -> Case {
    let (h, hkv) = (8, 2); // Qwen3.5-2B's full-attention heads
    let mut c = Case {
        label: label.into(),
        batch,
        tq,
        q_off,
        tkv,
        cap,
        h,
        hkv,
        scale,
        q: random_f32(batch * tq * h * D, seed).iter().map(|x| x * s_qk).collect(),
        k: random_f32(batch * cap * hkv * D, seed + 1)
            .iter()
            .map(|x| x * s_qk)
            .collect(),
        v: random_f32(batch * cap * hkv * D, seed + 2),
        rows: sample_rows(tq),
        tier1: s_qk <= 1.0,
    };
    c.poison_unread_kv(tkv);
    c
}

/// Every row of a short call; for a long one the block edges, the diagonal's
/// neighbours and the tail, which is where the masks and the ragged tile bite.
fn sample_rows(tq: usize) -> Vec<usize> {
    if tq <= 256 {
        return (0..tq).collect();
    }
    let mut rows: Vec<usize> = vec![0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 511, 512, 513, tq / 2];
    rows.extend(tq.saturating_sub(70)..tq);
    rows.retain(|&r| r < tq);
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The oracle's answer for one (batch, row, head).
struct RowRef {
    /// Offset of the row's first element in the output.
    at: usize,
    out: Vec<f64>,
    /// `A_x = sum_j w_j |v_jx|`.
    a: Vec<f64>,
    /// `max_j sum_i |q_i k_ji|` over the live keys.
    dq: f64,
    /// `max_j (m - s_j)`, the scaled score spread.
    spread: f64,
    live: usize,
}

fn oracle(c: &Case, q: &[f32], k: &[f32], v: &[f32]) -> Vec<RowRef> {
    let group = c.h / c.hkv;
    let mut refs = Vec::new();
    for b in 0..c.batch {
        for &t in &c.rows {
            for hd in 0..c.h {
                let hk = hd / group;
                let at = ((b * c.tq + t) * c.h + hd) * D;
                let live = c.tkv.min(c.q_off + t + 1);
                let kv = |j: usize| ((b * c.cap + j) * c.hkv + hk) * D;
                let mut scores = Vec::with_capacity(live);
                let mut dq = 0.0f64;
                for j in 0..live {
                    let (mut dot, mut abs) = (0.0f64, 0.0f64);
                    for x in 0..D {
                        let p = q[at + x] as f64 * k[kv(j) + x] as f64;
                        dot += p;
                        abs += p.abs();
                    }
                    scores.push(dot * c.scale as f64);
                    dq = dq.max(abs);
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let spread = scores.iter().map(|s| m - s).fold(0.0, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let l: f64 = w.iter().sum();
                let (mut out, mut a) = (vec![0.0; D], vec![0.0; D]);
                for (j, wj) in w.iter().enumerate() {
                    for x in 0..D {
                        let vv = v[kv(j) + x] as f64;
                        out[x] += wj / l * vv;
                        a[x] += wj / l * vv.abs();
                    }
                }
                refs.push(RowRef {
                    at,
                    out,
                    a,
                    dq,
                    spread,
                    live,
                });
            }
        }
    }
    refs
}

/// The derived bound on `|kernel - oracle|` at column `x` of a row. See the
/// module docs. `tier1` includes the rounding of the inputs to bf16.
fn bound(r: &RowRef, scale: f64, tier1: bool, x: usize) -> f64 {
    let d_in = if tier1 { (2.0 * U + U * U) * scale * r.dq } else { 0.0 };
    let d_acc = D as f64 * F32_EPS * scale * r.dq;
    let d_exp = (4.0 + 2.0 * r.spread) * F32_EPS;
    let a = d_in + U + d_acc + d_exp;
    let u_v = if tier1 { U } else { 0.0 };
    let g = (r.live + D) as f64 * F32_EPS;
    ((2.0 * a).exp_m1() * (1.0 + u_v) + u_v + g) * r.a[x]
}

/// Run one variant; the f32 output, or with `out_bf16` the raw bf16 bits packed
/// two to an f32 word.
fn run(rt: &std::sync::Arc<tessl::GpuRuntime>, c: &Case, tile: AttnTile, p: AttnProbTile, out_bf16: bool) -> Vec<f32> {
    let (qb, kb, vb) = (buf_bf16(rt, &c.q), buf_bf16(rt, &c.k), buf_bf16(rt, &c.v));
    let (tkv, q_off, kv_off) = (
        buf_u32(rt, &[c.tkv as u32]),
        buf_u32(rt, &[c.q_off as u32]),
        buf_u32(rt, &[0]),
    );
    let o = if out_bf16 {
        let o = rt.alloc_buffer(c.q_len() * 2).unwrap();
        o.write_bf16_bits(&vec![0xFFFF; c.q_len()]);
        o
    } else {
        seeded(rt, c.q_len(), UNWRITTEN)
    };
    let dims = AttnDims {
        batch: c.batch as u32,
        tq: c.tq as u32,
        heads: c.h as u32,
        heads_kv: c.hkv as u32,
        window: 0,
        scale: c.scale,
    };
    attn_prefill_bf16_with_tile(rt, &qb, &kb, &vb, &o, &tkv, &q_off, &kv_off, dims, out_bf16, tile, p)
        .unwrap_or_else(|e| panic!("{} {tile:?} {p:?}: {e}", c.label));
    rt.synchronize().unwrap();
    o.read_f32()
}

/// Check one variant's output against the oracles; the largest error over its
/// bound (at most 1 passes) per tier.
fn check(c: &Case, tile: AttnTile, p: AttnProbTile, got: &[f32], t1: &[RowRef], t2: &[RowRef]) -> (f64, f64) {
    let ctx = format!("{} tile {tile:?} {p:?}", c.label);
    assert_eq!(got.len(), c.q_len(), "{ctx}: output length");
    if let Some(i) = got.iter().position(|x| !x.is_finite() || *x == UNWRITTEN) {
        panic!("{ctx}: element {i} is unwritten or non-finite ({})", got[i]);
    }
    let mut worst = (0.0f64, 0.0f64);
    for (tier1, refs, slot) in [(true, t1, 0usize), (false, t2, 1)] {
        if refs.is_empty() {
            continue;
        }
        for r in refs {
            for x in 0..D {
                let err = (got[r.at + x] as f64 - r.out[x]).abs();
                let b = bound(r, c.scale as f64, tier1, x);
                assert!(
                    err <= b,
                    "{ctx}, tier {}: element {} (row offset {}, column {x}) got {} want {} err {err:.3e} > bound {b:.3e}",
                    if tier1 { 1 } else { 2 },
                    r.at + x,
                    r.at,
                    got[r.at + x],
                    r.out[x]
                );
                let ratio = if b > 0.0 { err / b } else { 0.0 };
                if slot == 0 {
                    worst.0 = worst.0.max(ratio);
                } else {
                    worst.1 = worst.1.max(ratio);
                }
            }
        }
    }
    worst
}

fn run_case(rt: &std::sync::Arc<tessl::GpuRuntime>, c: &Case) {
    let t1 = if c.tier1 {
        oracle(c, &c.q, &c.k, &c.v)
    } else {
        Vec::new()
    };
    let (qr, kr, vr) = (round_trip_bf16(&c.q), round_trip_bf16(&c.k), round_trip_bf16(&c.v));
    let t2 = oracle(c, &qr, &kr, &vr);
    for (tile, p) in variants() {
        let got = run(rt, c, tile, p, false);
        let (w1, w2) = check(c, tile, p, &got, &t1, &t2);
        eprintln!(
            "{:<34} {:<9} {:?}: max err/bound tier1 {w1:.3} tier2 {w2:.3}",
            c.label,
            format!("{tile:?}"),
            p
        );
    }
}

/// The Qwen3.5 prefill shapes: 8 query heads on 2 KV heads of 256 (read from
/// `docs/qwen35.md` and `bench_qwen35_layers`'s `Q_HEADS`/`KV_HEADS`/`HEAD_DIM`),
/// the model's score scale `256^-1/2`, causal. Lengths cover one query, tails
/// shorter than a tile, a tile boundary +-1, and a ragged multi-block call; the
/// last is a long one checked at the rows where the masks bite.
#[test]
fn bf16_tiles_match_the_f64_oracle_across_prefill_shapes() {
    with_gpu(|rt| {
        let scale = 1.0 / 16.0;
        for (i, t) in [1usize, 7, 31, 33, 65, 200].into_iter().enumerate() {
            run_case(
                rt,
                &random_case(
                    &format!("moderate T={t}"),
                    1,
                    t,
                    0,
                    t,
                    t,
                    scale,
                    1.0,
                    0xA000 + 8 * i as u64,
                ),
            );
        }
        run_case(
            rt,
            &random_case(
                "moderate T=1030 (sampled rows)",
                1,
                1030,
                0,
                1030,
                1030,
                scale,
                1.0,
                0xA100,
            ),
        );
        // Chunked prefill: queries at 100.., a cache with 160 slots of which 120
        // are live; the rest hold 3e4 and must never be read.
        run_case(
            rt,
            &random_case(
                "chunk q_off=100 tq=40 tkv=120 cap=160",
                1,
                40,
                100,
                120,
                160,
                scale,
                1.0,
                0xA200,
            ),
        );
        run_case(
            rt,
            &random_case("batch=2 T=45 cap=64", 2, 45, 0, 45, 64, scale, 1.0, 0xA300),
        );
        // No live key: every row is zeros, and the bound is zero there.
        run_case(rt, &random_case("tkv=0", 1, 5, 0, 0, 8, scale, 1.0, 0xA400));
    });
}

/// Scores with a standard deviation near 5 (the existing tiled test's regime):
/// the online softmax rescales between blocks and P is far from uniform. Only
/// tier 2: the input rounding moves a score by ~0.25 here, which is the real
/// sensitivity of a sharp softmax, and tier 1's bound is correspondingly loose.
#[test]
fn bf16_tiles_match_the_oracle_when_scores_are_sharp_across_blocks() {
    with_gpu(|rt| {
        run_case(
            rt,
            &random_case("strong scores T=200", 1, 200, 0, 200, 200, 0.25, 2.0, 0xB000),
        );
    });
}

/// A sharp row: one key's scaled score is 60 above every other's, so the row's
/// output is that key's V row. The row maximum is exactly 0 after the subtraction,
/// `exp(0) = 1` exactly (and bf16 holds 1 exactly), every other weight is below
/// 1e-25, and the row sum is exactly 1: the bf16 entries return the bf16-rounded V
/// row to the last bit. (The f32 entry's P is S in place, so there `exp(s - m)`
/// is the fast-math `exp` of a value that is 0 up to the compiler's contraction of
/// `s * scale - m`; the result agrees with V to f32 rounding.)
///
/// Also pinned for the all-equal row below. The current f32 body already satisfies
/// the property (`attn_tiled.h` scales first, takes the max of the scaled scores
/// and subtracts it), so there is no failing case to invent for it.
#[test]
fn the_row_maximum_contributes_exactly_one_on_a_sharp_row() {
    with_gpu(|rt| {
        let (t, key) = (100usize, 5usize);
        let mut c = random_case("sharp row", 1, t, 0, t, t, 1.0 / 16.0, 1.0, 0xC000);
        c.tier1 = false;
        let u = random_f32(D, 0xC100);
        let norm2: f32 = u.iter().map(|x| x * x).sum();
        // scaled score of `key` = scale * a * |u|^2 = 60.
        let a = 60.0 * 16.0 / norm2;
        let mut rng = SplitMix::new(0xC200);
        for b in 0..c.batch {
            for tq in 0..t {
                for hd in 0..c.h {
                    let at = ((b * c.tq + tq) * c.h + hd) * D;
                    c.q[at..at + D].copy_from_slice(&u);
                }
            }
            for j in 0..t {
                for hk in 0..c.hkv {
                    let at = ((b * c.cap + j) * c.hkv + hk) * D;
                    for (x, ux) in u.iter().enumerate() {
                        c.k[at + x] = if j == key { a * ux } else { 0.02 * rng.unit() };
                    }
                }
            }
        }
        let (qr, kr, vr) = (round_trip_bf16(&c.q), round_trip_bf16(&c.k), round_trip_bf16(&c.v));
        let refs = oracle(&c, &qr, &kr, &vr);
        for r in refs.iter().filter(|r| r.live > key) {
            assert!(r.spread > 50.0, "the case is not sharp: spread {}", r.spread);
        }
        let group = c.h / c.hkv;
        for (tile, p) in variants() {
            let got = run(rt, &c, tile, p, false);
            check(&c, tile, p, &got, &[], &refs);
            for b in 0..c.batch {
                for tq in key..t {
                    for hd in 0..c.h {
                        let at = ((b * c.tq + tq) * c.h + hd) * D;
                        let vrow = ((b * c.cap + key) * c.hkv + hd / group) * D;
                        for x in 0..D {
                            let want = vr[vrow + x];
                            let tol = match p {
                                AttnProbTile::Bf16 => 1e-12,
                                AttnProbTile::F32 => 4.0 * f64::from(want.abs()) * F32_EPS + 1e-12,
                            };
                            let err = (f64::from(got[at + x]) - f64::from(want)).abs();
                            assert!(
                                err <= tol,
                                "sharp row, {tile:?} {p:?}: row {tq} head {hd} column {x} got {} want {want} (err {err:.3e}, tol {tol:.3e})",
                                got[at + x]
                            );
                        }
                    }
                }
            }
        }
    });
}

/// An all-equal row: every key is the same vector, so every score of a row is
/// the same number (a nonzero one) and every weight is 1; the output is the mean
/// of the live V rows. Within the derived bound, for every tile.
#[test]
fn an_all_equal_row_returns_the_mean_of_v() {
    with_gpu(|rt| {
        let t = 97usize;
        let mut c = random_case("all-equal scores", 1, t, 0, t, t, 1.0 / 16.0, 1.0, 0xD000);
        let u = random_f32(D, 0xD100);
        let w: Vec<f32> = random_f32(D, 0xD200).iter().map(|x| 0.5 * x).collect();
        for tq in 0..t {
            for hd in 0..c.h {
                let at = (tq * c.h + hd) * D;
                c.q[at..at + D].copy_from_slice(&u);
            }
        }
        for j in 0..t {
            for hk in 0..c.hkv {
                let at = (j * c.hkv + hk) * D;
                c.k[at..at + D].copy_from_slice(&w);
            }
        }
        let dot: f32 = u.iter().zip(&w).map(|(a, b)| a * b).sum();
        assert!(
            dot.abs() > 1.0,
            "the shared score is {dot}: too close to zero to be a nonzero all-equal row"
        );
        run_case(rt, &c);
    });
}

/// The kernel is deterministic (no atomics, a fixed accumulation order), and
/// `out_bf16` stores exactly the round-to-nearest-even of the f32 output.
#[test]
fn out_bf16_is_the_rounded_f32_output_and_runs_repeat_bit_for_bit() {
    with_gpu(|rt| {
        let c = random_case("determinism T=200", 1, 200, 0, 200, 200, 1.0 / 16.0, 1.0, 0xE000);
        for (tile, p) in variants() {
            let a = run(rt, &c, tile, p, false);
            let b = run(rt, &c, tile, p, false);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{tile:?} {p:?}: two runs differ"
            );
            let packed = run(rt, &c, tile, p, true);
            for (i, f) in a.iter().enumerate() {
                let word = packed[i / 2].to_bits();
                let bits = if i % 2 == 0 { word & 0xFFFF } else { word >> 16 } as u16;
                assert_eq!(
                    bits,
                    f32_to_bf16_bits_gpu(*f),
                    "{tile:?} {p:?}: element {i}, f32 output {f}"
                );
            }
        }
    });
}

/// The host entry refuses what the kernels cannot do, before any dispatch.
#[test]
fn the_host_entry_refuses_what_the_kernels_cannot_do() {
    with_gpu(|rt| {
        let c = random_case("refusals", 1, 8, 0, 8, 8, 1.0 / 16.0, 1.0, 0xF000);
        let (qb, kb, vb) = (buf_bf16(rt, &c.q), buf_bf16(rt, &c.k), buf_bf16(rt, &c.v));
        let o = buf(rt, &vec![0.0; c.q_len()]);
        let (tkv, zero) = (buf_u32(rt, &[8]), buf_u32(rt, &[0]));
        let dims = AttnDims {
            batch: 1,
            tq: 8,
            heads: 8,
            heads_kv: 2,
            window: 0,
            scale: 0.0625,
        };
        let go = |dims: AttnDims, q: &tessl::tensor::GpuBuffer, o: &tessl::tensor::GpuBuffer, tile, p| {
            attn_prefill_bf16_with_tile(rt, q, &kb, &vb, o, &tkv, &zero, &zero, dims, false, tile, p)
        };
        let ok = AttnTile::Q32K32Sg4;
        assert!(go(dims, &qb, &o, ok, AttnProbTile::Bf16).is_ok());
        let windowed = AttnDims { window: 4, ..dims };
        assert!(go(windowed, &qb, &o, ok, AttnProbTile::Bf16)
            .unwrap_err()
            .contains("window must be 0"));
        let ragged_groups = AttnDims { heads: 7, ..dims };
        assert!(go(ragged_groups, &qb, &o, ok, AttnProbTile::Bf16)
            .unwrap_err()
            .contains("multiple of heads_kv"));
        let bad_scale = AttnDims {
            scale: f32::NAN,
            ..dims
        };
        assert!(go(bad_scale, &qb, &o, ok, AttnProbTile::Bf16)
            .unwrap_err()
            .contains("scale must be finite"));
        // An f32-sized Q passes as bf16 only if it is big enough: half the elements is refused.
        let short_q = buf_bf16(rt, &c.q[..c.q_len() / 2]);
        assert!(go(dims, &short_q, &o, ok, AttnProbTile::Bf16)
            .unwrap_err()
            .contains("buffer holds"));
        // The output may not alias an input.
        assert!(go(dims, &qb, &qb, ok, AttnProbTile::Bf16).is_err());
        // The f32-P entry exists at one geometry only.
        assert!(go(dims, &qb, &o, AttnTile::Q64K64Sg8, AttnProbTile::F32)
            .unwrap_err()
            .contains("compiled only at"));
        // No work is a no-op, not an error.
        let empty = AttnDims { tq: 0, ..dims };
        assert!(go(empty, &qb, &o, ok, AttnProbTile::Bf16).is_ok());
    });
}

/// Device-free: each entry name spells the geometry its instantiation has, every
/// pair the host can name is instantiated in the metallib source, and the f32-P
/// variant exists at exactly one geometry.
#[test]
fn entry_names_spell_the_instantiated_geometry() {
    let mut named = 0;
    for (tile, p) in variants() {
        let entry = attn_prefill_bf16_entry(tile, p).unwrap();
        let (bq, bk, sg) = tile.geometry();
        assert!(
            entry.contains(&format!("_h256_{}", tile.label())),
            "{entry} does not spell {}",
            tile.label()
        );
        let macro_name = match p {
            AttnProbTile::Bf16 => "TILED_ATTN_BF16_KERNEL",
            AttnProbTile::F32 => "TILED_ATTN_BF16_PF32_KERNEL",
        };
        let line = format!("{macro_name}({entry}, {bq}, {bk}, {sg})");
        assert!(METAL.contains(&line), "kernels/qwen35_attn_tiled.metal lacks `{line}`");
        named += 1;
    }
    assert_eq!(named, 5);
    for tile in AttnTile::ALL {
        let f32p = attn_prefill_bf16_entry(tile, AttnProbTile::F32);
        assert_eq!(f32p.is_ok(), tile == AttnTile::Q32K32Sg4, "{tile:?}");
        assert!(attn_prefill_bf16_entry(tile, AttnProbTile::Bf16).is_ok());
    }
    // Nothing is instantiated that the host cannot name.
    let instantiated =
        METAL.matches("\nTILED_ATTN_BF16_KERNEL(").count() + METAL.matches("\nTILED_ATTN_BF16_PF32_KERNEL(").count();
    assert_eq!(instantiated, named);
}

/// Device-free: the oracle is not a tautology. A two-key row with scores 0 and
/// ln 3 has weights 1/4 and 3/4 by hand, and the oracle and its bound agree.
#[test]
fn the_oracle_reproduces_a_two_key_row_by_hand() {
    let (h, hkv, scale) = (1usize, 1usize, 0.5f32);
    let mut q = vec![0.0f32; D];
    q[0] = 1.0;
    let mut k = vec![0.0f32; 2 * D];
    k[D] = 3.0f32.ln() / scale; // key 1: score ln 3; key 0: score 0
    let mut v = vec![0.0f32; 2 * D];
    v[0] = 4.0;
    v[D] = -8.0;
    let c = Case {
        label: "hand".into(),
        batch: 1,
        tq: 1,
        q_off: 1, // sees keys 0 and 1
        tkv: 2,
        cap: 2,
        h,
        hkv,
        scale,
        q: q.clone(),
        k: k.clone(),
        v: v.clone(),
        rows: vec![0],
        tier1: true,
    };
    let refs = oracle(&c, &q, &k, &v);
    assert_eq!(refs.len(), 1);
    let r = &refs[0];
    assert_eq!(r.live, 2);
    // weights 1/4 and 3/4: out = 4/4 - 8*3/4 = -5; A = 4/4 + 8*3/4 = 7.
    assert!((r.out[0] + 5.0).abs() < 1e-6, "{}", r.out[0]);
    assert!((r.a[0] - 7.0).abs() < 1e-6, "{}", r.a[0]);
    assert!((r.spread - 3.0f64.ln()).abs() < 1e-6, "{}", r.spread);
    // dq = |q.k| for key 1 = ln 3 / scale; the bound is positive, grows with
    // the input rounding, and is far below the output's own scale here.
    let (b1, b2) = (bound(r, scale as f64, true, 0), bound(r, scale as f64, false, 0));
    assert!(b1 > b2 && b2 > 0.0, "{b1} {b2}");
    assert!(b1 < 0.2 * r.a[0], "{b1}");
}
