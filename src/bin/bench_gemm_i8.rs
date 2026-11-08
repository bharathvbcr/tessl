//! Time the shipped int8 TensorOps path on `(M, 4096, 4096)`.
//!
//! `bench_gemm_sweep` times f32, tf32 and bf16. This binary times
//! `nn::gemm_i8` and the quantize-then-dequant pipeline
//! (`nn::quantize_i8_rows` plus per-row scales). MLX and MPS are not linked
//! here.
//!
//! Protocol: for each M the four arms below run interleaved
//! (`common::run_paired`), `BENCH_WARMUP` untimed calls then `BENCH_ROUNDS`
//! rounds of `BENCH_ITERS` samples, each sample `BENCH_CALLS` back-to-back
//! calls and one `synchronize`, reported in ms per call. Defaults: warmup 2,
//! iters 3, rounds 5, calls 4.
//!
//! The public entries drain the host on their own, which a timed region must
//! not hide: `quantize_i8_rows` reads back its non-finite flags after every
//! call, and `gemm_i8` with `I8AScale::Rows` maps the scale buffer to check it
//! is finite (`require_finite`), which waits for every queued command. The
//! `gemm-scalar` arm has no buffer to check, so its calls queue; the
//! `gemm-rows` arm differs from it only by that validation. Their per-round
//! ratio is printed as the drain's cost, and every arm is labelled with
//! whether its number includes a per-call host sync.
//!
//! Before any number is trusted, sampled positions of C (the four corners and
//! seeded draws) are checked against a CPU oracle for both the scalar and the
//! quantized path; a mismatch exits 1.

mod common;

use std::collections::HashSet;

use common::{env_usize, jarr, jnum, jstr, median, ratio_stats, run_paired, Arm, Lcg};
use tessl::nn::{self, I8AScale, I8BLayout, I8BScale, I8Gemm, I8Out};
use tessl::runtime::GpuRuntime;

const MS: [u32; 8] = [1, 16, 64, 128, 256, 512, 1024, 2048];
const N: u32 = 4096;
const K: u32 = 4096;
/// Seeded draws of C checked per M and path, on top of the corners, the tile
/// edges and one position in every tile.
const SAMPLES: usize = 64;
/// Columns per output tile of the int8 GEMM (`dispatch_i8`: 64 rows for
/// `m <= 64`, 128 otherwise, always 64 columns). The oracle mirrors it only to
/// choose where to look; a wrong guess narrows coverage, not correctness.
const TILE_N: usize = 64;
const SEED: u64 = 0x18_6e_70_5a;

fn main() {
    if let Err(err) = run() {
        eprintln!("bench_gemm_i8: {err}");
        std::process::exit(1);
    }
}

/// One timed arm: what it runs and whether its number includes a host sync
/// the call makes on its own (as opposed to the one `synchronize` per sample).
struct ArmInfo {
    name: &'static str,
    what: &'static str,
    host_sync_per_call: bool,
    flops: bool,
}

const ARMS: [ArmInfo; 4] = [
    ArmInfo {
        name: "tessl-i8-gemm",
        what: "nn::gemm_i8 scalar a_scale 1, no b scale, f32 store, B [K,N]; nothing to validate, calls queue",
        host_sync_per_call: false,
        flops: true,
    },
    ArmInfo {
        name: "tessl-i8-gemm-rows",
        what: "nn::gemm_i8 with per-row a_scale; includes require_finite's per-call host sync (waits for queued work)",
        host_sync_per_call: true,
        flops: true,
    },
    ArmInfo {
        name: "tessl-i8-quantize",
        what: "nn::quantize_i8_rows alone; includes its per-call flag readback (host sync)",
        host_sync_per_call: true,
        flops: false,
    },
    ArmInfo {
        name: "tessl-i8-pipeline",
        what: "nn::quantize_i8_rows then nn::gemm_i8 with per-row a_scale; includes both per-call host syncs",
        host_sync_per_call: true,
        flops: true,
    },
];

fn run() -> Result<(), String> {
    let warmup = env_usize("BENCH_WARMUP", 2, 0)?;
    let iters = env_usize("BENCH_ITERS", 3, 1)?;
    let rounds = env_usize("BENCH_ROUNDS", 5, 1)?;
    let calls = env_usize("BENCH_CALLS", 4, 1)?;
    let rt = GpuRuntime::new()?;
    let device = rt.device_name();
    let m_max = 2048usize;
    let a = rt.alloc_buffer(m_max * K as usize)?;
    let b = rt.alloc_buffer((K as usize) * (N as usize))?;
    let c = rt.alloc_buffer(m_max * (N as usize) * 4)?;
    let x = rt.alloc_buffer(m_max * (K as usize) * 4)?;
    let q = rt.alloc_buffer(m_max * K as usize)?;
    let scales = rt.alloc_buffer(m_max * 4)?;

    let a_bytes = fill_a(m_max);
    a.try_write_bytes(&a_bytes)?;
    let b_bytes = fill_b();
    b.try_write_bytes(&b_bytes)?;
    let xf = fill_x(m_max);
    x.try_write_f32(&xf)?;

    // The row scales the `gemm-rows` arm reads: quantized once, untimed.
    nn::quantize_i8_rows(&rt, &x, &q, &scales, m_max as u32, K)?;
    rt.synchronize()?;

    let mut rows_by_arm: Vec<Vec<String>> = vec![Vec::new(); ARMS.len()];
    for &m in &MS {
        let pipeline = nn::i8_gemm_pipeline_name(m, I8BLayout::RowMajorKN, 0)?;
        verify_m(
            &rt,
            m,
            pipeline,
            &Bufs {
                a: &a,
                b: &b,
                c: &c,
                x: &x,
                q: &q,
                scales: &scales,
            },
            (&a_bytes, &b_bytes, &xf),
        )?;

        let reps = calls;
        let mut paired_arms = vec![
            Arm {
                name: ARMS[0].name.to_string(),
                reps,
                run: Box::new(|| {
                    for _ in 0..reps {
                        call_gemm(&rt, &a, &b, &c, m)?;
                    }
                    rt.synchronize()
                }),
            },
            Arm {
                name: ARMS[1].name.to_string(),
                reps,
                run: Box::new(|| {
                    for _ in 0..reps {
                        call_rows(&rt, &q, &b, &c, &scales, m)?;
                    }
                    rt.synchronize()
                }),
            },
            Arm {
                name: ARMS[2].name.to_string(),
                reps,
                run: Box::new(|| {
                    for _ in 0..reps {
                        nn::quantize_i8_rows(&rt, &x, &q, &scales, m, K)?;
                    }
                    rt.synchronize()
                }),
            },
            Arm {
                name: ARMS[3].name.to_string(),
                reps,
                run: Box::new(|| {
                    for _ in 0..reps {
                        nn::quantize_i8_rows(&rt, &x, &q, &scales, m, K)?;
                        call_rows(&rt, &q, &b, &c, &scales, m)?;
                    }
                    rt.synchronize()
                }),
            },
        ];
        let paired = run_paired(&mut paired_arms, warmup, iters, rounds)?;
        drop(paired_arms);
        let (drain, lo, hi) = ratio_stats(&paired.rounds[0], &paired.rounds[1])?;
        eprintln!("M={m} N={N} K={K} pipeline={pipeline}");
        for (info, per_round) in ARMS.iter().zip(&paired.rounds) {
            eprintln!("  {}", row(info, m, per_round)?);
        }
        eprintln!(
            "  per-call host sync of Rows validation: tessl-i8-gemm / tessl-i8-gemm-rows = {drain:.3} (rounds {lo:.3}..{hi:.3}), \
             {calls} calls queued per sample"
        );
        for (i, (info, per_round)) in ARMS.iter().zip(&paired.rounds).enumerate() {
            rows_by_arm[i].push(json_row(info, m, pipeline, per_round)?);
        }
    }

    println!("{{");
    println!("  \"entry\": \"bench_gemm_i8\",");
    println!("  \"device\": {},", jstr(&device));
    println!("  \"n\": {N},");
    println!("  \"k\": {K},");
    println!("  \"warmup\": {warmup},");
    println!("  \"repeat_count\": {iters},");
    println!("  \"rounds\": {rounds},");
    println!("  \"calls_per_sample\": {calls},");
    println!("  \"oracle_positions_per_m\": {SAMPLES},");
    println!("  \"oracle_checked\": true,");
    println!("  \"synchronize_every_sample\": true,");
    println!("  \"time_unit\": \"ms per call, median of per-round medians\",");
    println!("  \"runtime\": \"GpuRuntime::new (counter timestamps on)\",");
    println!("  \"arms\": [");
    for (i, info) in ARMS.iter().enumerate() {
        println!("    {{");
        println!("      \"name\": {},", jstr(info.name));
        println!("      \"what\": {},", jstr(info.what));
        println!("      \"includes_per_call_host_sync\": {},", info.host_sync_per_call);
        println!("      \"rows\": [");
        println!("        {}", rows_by_arm[i].join(",\n        "));
        println!("      ]");
        println!("    }}{}", if i + 1 < ARMS.len() { "," } else { "" });
    }
    println!("  ]");
    println!("}}");
    Ok(())
}

struct Bufs<'a> {
    a: &'a tessl::tensor::GpuBuffer,
    b: &'a tessl::tensor::GpuBuffer,
    c: &'a tessl::tensor::GpuBuffer,
    x: &'a tessl::tensor::GpuBuffer,
    q: &'a tessl::tensor::GpuBuffer,
    scales: &'a tessl::tensor::GpuBuffer,
}

/// Run both paths once at `m` and check sampled positions of C against the
/// CPU oracle. `host` is the host copy of (A bytes, B bytes, X floats).
fn verify_m(
    rt: &std::sync::Arc<GpuRuntime>,
    m: u32,
    pipeline: &str,
    bufs: &Bufs<'_>,
    host: (&[u8], &[u8], &[f32]),
) -> Result<(), String> {
    let (a_bytes, b_bytes, xf) = host;
    let (mu, nu, ku) = (m as usize, N as usize, K as usize);
    let positions = sample_positions(mu, nu, SEED ^ u64::from(m), SAMPLES, (tile_rows(mu), TILE_N));

    call_gemm(rt, bufs.a, bufs.b, bufs.c, m)?;
    rt.synchronize()?;
    {
        let c = bufs.c.try_contents_f32()?;
        check_sampled(
            &format!("gemm_i8 {pipeline} M={m}"),
            &c[..mu * nu],
            &a_bytes[..mu * ku],
            b_bytes,
            (nu, ku),
            &|_| 1.0,
            &positions,
        )?;
    }

    nn::quantize_i8_rows(rt, bufs.x, bufs.q, bufs.scales, m, K)?;
    call_rows(rt, bufs.q, bufs.b, bufs.c, bufs.scales, m)?;
    rt.synchronize()?;
    let scales = bufs.scales.try_read_f32()?;
    // One host mapping at a time (ft-4b1313de: the runtime lease is exclusive,
    // a second live mapping fails "runtime busy"): copy q's rows out first.
    let q = bufs.q.try_contents_u8()?[..mu * ku].to_vec();
    let rows: std::collections::BTreeSet<usize> = positions.iter().map(|&(r, _)| r).collect();
    for &row in &rows {
        check_quantized(
            &format!("quantize_i8_rows {pipeline} M={m} row {row}"),
            &xf[row * ku..(row + 1) * ku],
            &q[row * ku..(row + 1) * ku],
            scales[row],
        )?;
    }
    let c = bufs.c.try_contents_f32()?;
    check_sampled(
        &format!("quantize+gemm_i8 {pipeline} M={m}"),
        &c[..mu * nu],
        &q,
        b_bytes,
        (nu, ku),
        &|row| scales[row],
        &positions,
    )
}

/// Rows per output tile for an `m`-row call (see [`TILE_N`]).
fn tile_rows(m: usize) -> usize {
    if m <= 64 {
        64
    } else {
        128
    }
}

/// SplitMix64: operands with no period, so no two rows of A or columns of B
/// repeat (a period dividing K made every row identical).
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A, `m_max x K` int8 as bytes, every row distinct.
fn fill_a(m_max: usize) -> Vec<u8> {
    (0..(m_max * K as usize) as u64)
        .map(|i| mix(0xA ^ (i << 8)) as u8)
        .collect()
}

/// B, `K x N` int8 as bytes, every row and column distinct.
fn fill_b() -> Vec<u8> {
    (0..(K as u64 * u64::from(N)))
        .map(|i| mix(0xB ^ (i << 8)) as u8)
        .collect()
}

/// Row `r`'s amplitude: distinct per row, so every row scale differs.
fn amplitude(r: usize) -> f32 {
    0.25 + 0.001 * r as f32
}

/// X, `m_max x K` floats: per-row amplitude times a seeded draw in [-1, 1],
/// with `x[r][0]` pinned to the amplitude so row `r`'s absmax is exactly
/// `amplitude(r)` and its scale is `amplitude(r) / 127`.
fn fill_x(m_max: usize) -> Vec<f32> {
    let k = K as usize;
    let mut x = Vec::with_capacity(m_max * k);
    for r in 0..m_max {
        for c in 0..k {
            let draw = (mix(0xC ^ (((r * k + c) as u64) << 8)) % 2001) as f32;
            x.push(if c == 0 {
                amplitude(r)
            } else {
                amplitude(r) * ((draw - 1000.0) / 1000.0)
            });
        }
    }
    x
}

/// Check one quantized row against the host's IEEE quantize, within exactly the
/// tolerance `src/i8_pipeline.rs` `quantize_i8_rows` documents (da0bd11) and
/// `tests/gemm_i8.rs` asserts: the kernel is fast-math, so
/// - the scale is within 1 ulp of the host's `absmax / 127` (1 for a zero row);
/// - `q` is compared with `clamp(rint(r), -127, 127)` for `r = x / scale`, using
///   the scale the kernel wrote, and may differ by 1 only where
///   `|r - (floor(r) + 0.5)| <= 8 * f32::EPSILON * |r|`; everywhere else it
///   must be exact. `floor` for `rint`, or a wrong scale, fails.
fn check_quantized(label: &str, x: &[f32], q: &[u8], scale: f32) -> Result<(), String> {
    let want_scale = row_scale(x);
    let ulps = (i64::from(scale.to_bits()) - i64::from(want_scale.to_bits())).abs();
    if ulps > 1 {
        return Err(format!(
            "{label}: scale {scale} against host {want_scale}: {ulps} ulp apart (at most 1 allowed)"
        ));
    }
    for (i, (&xv, &qb)) in x.iter().zip(q).enumerate() {
        let r = xv / scale;
        let want = r.round_ties_even().clamp(-127.0, 127.0) as i32;
        let got = i32::from(qb as i8);
        let near_tie = (r - (r.floor() + 0.5)).abs() <= 8.0 * f32::EPSILON * r.abs();
        if got != want && !(near_tie && (got - want).abs() == 1) {
            return Err(format!(
                "{label}: q[{i}] = {got}, host quantize of {xv} / {scale} = {r} gives {want}"
            ));
        }
    }
    Ok(())
}

/// `absmax / 127`, or 1 for an all-zero row: `quantize_i8_rows`' contract.
fn row_scale(row: &[f32]) -> f32 {
    let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    if absmax == 0.0 {
        1.0
    } else {
        absmax / 127.0
    }
}

/// Positions of an `m x n` matrix to check, in order, without repeats: its
/// four corners; the cells either side of tile boundaries (last row/column of a
/// tile and first of the next, at the first and last boundary); one seeded cell
/// inside every `tile.0 x tile.1` tile, so a skipped tile cannot hide; then
/// seeded draws until `draws` of them have been added.
fn sample_positions(m: usize, n: usize, seed: u64, draws: usize, tile: (usize, usize)) -> Vec<(usize, usize)> {
    let (tm, tn) = tile;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut add = |p: (usize, usize), out: &mut Vec<(usize, usize)>| {
        if p.0 < m && p.1 < n && seen.insert(p) {
            out.push(p);
        }
    };
    for p in [(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)] {
        add(p, &mut out);
    }
    let edges = |extent: usize, t: usize| -> Vec<usize> {
        let last = (extent - 1) / t * t;
        let mut v = vec![t - 1, t, last.saturating_sub(1), last];
        v.sort_unstable();
        v.dedup();
        v
    };
    for &r in &edges(m, tm) {
        for &c in &edges(n, tn) {
            add((r, c), &mut out);
        }
    }
    let mut rng = Lcg::new(seed);
    let mut pick = |lo: usize, len: usize| lo + ((rng.next_u01() * len as f64) as usize).min(len - 1);
    for ty in 0..m.div_ceil(tm) {
        for tx in 0..n.div_ceil(tn) {
            let r = pick(ty * tm, tm.min(m - ty * tm));
            let c = pick(tx * tn, tn.min(n - tx * tn));
            add((r, c), &mut out);
        }
    }
    for _ in 0..draws {
        let p = (pick(0, m), pick(0, n));
        add(p, &mut out);
    }
    out
}

/// `sum_k a[row,k] * b[k,col]` over signed bytes, exact, and the sum of the
/// magnitudes of the terms (what a float accumulation's error scales with).
fn oracle_dot(a: &[u8], b: &[u8], (n, k): (usize, usize), row: usize, col: usize) -> (i64, f64) {
    let (mut sum, mut mag) = (0i64, 0f64);
    for i in 0..k {
        let t = i64::from(a[row * k + i] as i8) * i64::from(b[i * n + col] as i8);
        sum += t;
        mag += t.abs() as f64;
    }
    (sum, mag)
}

/// Check `c[row * n + col]` at each position against the oracle times that
/// row's scale. The tolerance is 1e-6 of the term magnitudes plus 1e-6 of the
/// result: a float accumulation stays inside it, a misplaced tile or a wrong
/// operand (errors on the order of the result) does not. A non-finite value
/// fails.
fn check_sampled(
    label: &str,
    c: &[f32],
    a: &[u8],
    b: &[u8],
    (n, k): (usize, usize),
    scale: &dyn Fn(usize) -> f32,
    positions: &[(usize, usize)],
) -> Result<(), String> {
    for &(row, col) in positions {
        let (sum, mag) = oracle_dot(a, b, (n, k), row, col);
        let s = f64::from(scale(row));
        let want = sum as f64 * s;
        let got = f64::from(c[row * n + col]);
        let tol = 1e-6 * mag * s.abs() + 1e-6 * want.abs();
        if !got.is_finite() || (got - want).abs() > tol {
            return Err(format!(
                "{label}: C[{row},{col}] = {got} against oracle {want} (tolerance {tol:e})"
            ));
        }
    }
    Ok(())
}

fn call_gemm(
    rt: &std::sync::Arc<GpuRuntime>,
    a: &tessl::tensor::GpuBuffer,
    b: &tessl::tensor::GpuBuffer,
    c: &tessl::tensor::GpuBuffer,
    m: u32,
) -> Result<(), String> {
    nn::gemm_i8(
        rt,
        I8Gemm {
            a,
            b,
            c,
            m,
            n: N,
            k: K,
            a_scale: I8AScale::Scalar(1.0),
            b_scale: I8BScale::None,
            bias: None,
            b_layout: I8BLayout::RowMajorKN,
            out: I8Out::F32,
        },
    )
}

fn call_rows(
    rt: &std::sync::Arc<GpuRuntime>,
    a: &tessl::tensor::GpuBuffer,
    b: &tessl::tensor::GpuBuffer,
    c: &tessl::tensor::GpuBuffer,
    scales: &tessl::tensor::GpuBuffer,
    m: u32,
) -> Result<(), String> {
    nn::gemm_i8(
        rt,
        I8Gemm {
            a,
            b,
            c,
            m,
            n: N,
            k: K,
            a_scale: I8AScale::Rows(scales),
            b_scale: I8BScale::None,
            bias: None,
            b_layout: I8BLayout::RowMajorKN,
            out: I8Out::F32,
        },
    )
}

/// One printed line for an arm: median over rounds, in ms per call.
fn row(info: &ArmInfo, m: u32, per_round: &[f64]) -> Result<String, String> {
    let med = median(per_round.to_vec())?;
    let tail = if info.flops {
        format!("  {:.1} GFLOP/s", gflops(m, med))
    } else {
        String::new()
    };
    let sync = if info.host_sync_per_call {
        "includes per-call host sync"
    } else {
        "no per-call host sync"
    };
    Ok(format!(
        "{} {med:.3} ms{tail}  ({sync})  rounds {}",
        info.name,
        jarr(per_round)
    ))
}

fn gflops(m: u32, ms: f64) -> f64 {
    (2.0 * f64::from(m) * f64::from(N) * f64::from(K)) / (ms * 1.0e6)
}

fn json_row(info: &ArmInfo, m: u32, pipeline: &str, per_round: &[f64]) -> Result<String, String> {
    let med = median(per_round.to_vec())?;
    let best = per_round.iter().copied().fold(f64::INFINITY, f64::min);
    let gf = if info.flops {
        jnum(gflops(m, med))
    } else {
        "null".to_string()
    };
    Ok(format!(
        "{{\"arm\": {}, \"m\": {m}, \"n\": {N}, \"k\": {K}, \"pipeline\": {}, \"median_ms\": {}, \"min_ms\": {}, \"gflops\": {gf}, \"round_medians_ms\": {}}}",
        jstr(info.name),
        jstr(pipeline),
        jnum(med),
        jnum(best),
        jarr(per_round)
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        amplitude, check_quantized, check_sampled, fill_a, fill_b, fill_x, oracle_dot, row_scale, sample_positions,
        tile_rows, K, N,
    };

    #[test]
    fn sample_positions_are_in_range_unique_seeded_and_start_with_the_corners() {
        let p = sample_positions(7, 5, 1, 64, (4, 2));
        assert_eq!(&p[..4], &[(0, 0), (0, 4), (6, 0), (6, 4)]);
        assert!(p.iter().all(|&(r, c)| r < 7 && c < 5));
        assert_eq!(p.len(), p.iter().collect::<std::collections::HashSet<_>>().len());
        assert_eq!(p, sample_positions(7, 5, 1, 64, (4, 2)));
        assert_ne!(p, sample_positions(7, 5, 2, 8, (4, 2)));
        // M = 1 stays inside the one row.
        assert!(sample_positions(1, 5, 1, 64, (4, 2)).iter().all(|&(r, _)| r == 0));
    }

    /// Every tile has a sampled position, and the cells either side of a tile
    /// boundary are sampled: a skipped tile or a tile placed one cell off has
    /// nowhere to hide.
    #[test]
    fn sample_positions_cover_every_tile_and_both_sides_of_the_edges() {
        let (m, n, tile) = (300, 200, (128, 64));
        let p = sample_positions(m, n, 3, 0, tile);
        for ty in 0..m.div_ceil(tile.0) {
            for tx in 0..n.div_ceil(tile.1) {
                assert!(
                    p.iter().any(|&(r, c)| r / tile.0 == ty && c / tile.1 == tx),
                    "no position in tile ({ty}, {tx})"
                );
            }
        }
        for edge in [(127, 63), (127, 64), (128, 63), (128, 64)] {
            assert!(p.contains(&edge), "{edge:?} not sampled");
        }
        assert_eq!(tile_rows(64), 64);
        assert_eq!(tile_rows(65), 128);
    }

    /// The old operands had period 256 in A (and in B's columns), so every row
    /// of A was identical, every row of B was identical and every row of X had
    /// the same absmax: a per-row or per-k indexing bug could not show.
    #[test]
    fn operands_differ_by_row_and_scales_differ_by_row() {
        let k = K as usize;
        let a = fill_a(512);
        assert_ne!(a[..k], a[256 * k..257 * k]);
        assert_ne!(a[..k], a[k..2 * k]);
        let b = fill_b();
        let n = N as usize;
        assert_ne!(b[..n], b[n..2 * n]);
        let x = fill_x(2048);
        let scales: Vec<u32> = (0..2048).map(|r| row_scale(&x[r * k..(r + 1) * k]).to_bits()).collect();
        let distinct: std::collections::HashSet<_> = scales.iter().collect();
        assert_eq!(distinct.len(), 2048);
        assert_eq!(row_scale(&x[..k]), amplitude(0) / 127.0);
        assert!(x[..k].iter().skip(1).any(|v| *v < 0.0) && x[..k].iter().any(|v| *v > 0.0));
    }

    #[test]
    fn oracle_reads_bytes_as_signed_and_b_as_k_by_n() {
        // a = [[1, -1]], b = [[2, 3], [255 (-1), 4]] (K = 2, N = 2).
        let (a, b) = ([1u8, 255], [2u8, 3, 255, 4]);
        assert_eq!(oracle_dot(&a, &b, (2, 2), 0, 0), (3, 3.0));
        assert_eq!(oracle_dot(&a, &b, (2, 2), 0, 1), (-1, 7.0));
    }

    #[test]
    fn row_scale_is_absmax_over_127_or_one_for_zero_rows() {
        assert_eq!(row_scale(&[0.0, 0.0]), 1.0);
        assert_eq!(row_scale(&[1.0, -127.0]), 1.0);
    }

    fn tiny() -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        // M = 2, K = 2, N = 2: C = A @ B scaled by 0.5 and 2.0 per row.
        let a = vec![1u8, 255, 3, 4];
        let b = vec![2u8, 3, 255, 4];
        let c = vec![1.5, -0.5, 2.0 * (6.0 - 4.0), 2.0 * (9.0 + 16.0)];
        (a, b, c)
    }

    #[test]
    fn check_sampled_accepts_the_oracle_and_rounding() {
        let (a, b, mut c) = tiny();
        let all = [(0, 0), (0, 1), (1, 0), (1, 1)];
        let scale = |row: usize| if row == 0 { 0.5 } else { 2.0 };
        check_sampled("t", &c, &a, &b, (2, 2), &scale, &all).unwrap();
        c[3] *= 1.0 + 1e-8;
        check_sampled("t", &c, &a, &b, (2, 2), &scale, &all).unwrap();
    }

    #[test]
    fn check_sampled_rejects_a_wrong_nan_or_misplaced_value() {
        let (a, b, c) = tiny();
        let all = [(0, 0), (0, 1), (1, 0), (1, 1)];
        let scale = |row: usize| if row == 0 { 0.5 } else { 2.0 };
        let mut wrong = c.clone();
        wrong[2] += 1.0;
        let err = check_sampled("t", &wrong, &a, &b, (2, 2), &scale, &all).unwrap_err();
        assert!(err.contains("C[1,0]"), "{err}");
        let mut nan = c.clone();
        nan[0] = f32::NAN;
        assert!(check_sampled("t", &nan, &a, &b, (2, 2), &scale, &all).is_err());
        // Tiles swapped (the failure a bad tile walk produces): rows exchanged.
        let swapped = vec![c[2], c[3], c[0], c[1]];
        assert!(check_sampled("t", &swapped, &a, &b, (2, 2), &scale, &all).is_err());
        // A wrong per-row scale is caught, not just a wrong product.
        assert!(check_sampled("t", &c, &a, &b, (2, 2), &|_| 0.5, &all).is_err());
    }

    /// The corpus the rejection tests share: M = 4, K = 2, N = 4, 2x2 tiles.
    fn grid() -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<(usize, usize)>) {
        let a = vec![1u8, 255, 3, 4, 5, 6, 7, 250];
        let b = vec![2u8, 3, 9, 1, 255, 4, 250, 8];
        let scale = |r: usize| 0.5 + r as f32;
        let mut c = Vec::new();
        for r in 0..4 {
            for col in 0..4 {
                let (sum, _) = oracle_dot(&a, &b, (4, 2), r, col);
                c.push(sum as f32 * scale(r));
            }
        }
        (a, b, c, sample_positions(4, 4, 1, 0, (2, 2)))
    }

    fn check_grid(
        c: &[f32],
        a: &[u8],
        b: &[u8],
        pos: &[(usize, usize)],
        scale: &dyn Fn(usize) -> f32,
    ) -> Result<(), String> {
        check_sampled("grid", c, a, b, (4, 2), scale, pos)
    }

    #[test]
    fn the_oracle_rejects_a_swapped_row() {
        let (a, b, mut c, pos) = grid();
        let scale = |r: usize| 0.5 + r as f32;
        check_grid(&c, &a, &b, &pos, &scale).unwrap();
        for col in 0..4 {
            c.swap(col, 4 + col);
        }
        assert!(check_grid(&c, &a, &b, &pos, &scale).is_err());
    }

    #[test]
    fn the_oracle_rejects_a_wrong_per_row_scale() {
        let (a, b, c, pos) = grid();
        // The kernel read row r ^ 1's scale (a_rows[m ^ 1]).
        let scale = |r: usize| 0.5 + (r ^ 1) as f32;
        assert!(check_grid(&c, &a, &b, &pos, &scale).is_err());
        // The scale dropped to 1 / 126 of 1 / 127 (`/127` -> `/126`) is 0.8%.
        let (a, b, c, pos) = grid();
        let off = |r: usize| (0.5 + r as f32) * (127.0 / 126.0);
        assert!(check_grid(&c, &a, &b, &pos, &off).is_err());
    }

    #[test]
    fn the_oracle_rejects_a_skipped_tile() {
        let (a, b, mut c, pos) = grid();
        let scale = |r: usize| 0.5 + r as f32;
        // Tile (row 1, column 0) of 2x2: rows 2..4, columns 0..2 left unwritten.
        for r in 2..4 {
            for col in 0..2 {
                c[r * 4 + col] = 0.0;
            }
        }
        let err = check_grid(&c, &a, &b, &pos, &scale).unwrap_err();
        assert!(err.contains("C["), "{err}");
    }

    #[test]
    fn quantize_check_accepts_rint_and_a_near_tie_either_way() {
        let x = [1.0f32, 0.3, -0.7, 0.5 * 1.0 / 127.0, 2.5 / 127.0, -127.0 / 127.0];
        let scale = row_scale(&x);
        let rint: Vec<u8> = x.iter().map(|v| (v / scale).round_ties_even() as i8 as u8).collect();
        check_quantized("rint", &x, &rint, scale).unwrap();
        // 2.5 is a tie: 2 (even, rint) and 3 are both inside the window.
        let mut tie = rint.clone();
        tie[4] = (rint[4] as i8 + 1) as u8;
        check_quantized("tie", &x, &tie, scale).unwrap();
    }

    #[test]
    fn quantize_check_rejects_floor_for_rint_and_a_wrong_scale() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 31.3) * 0.0173).collect();
        let scale = row_scale(&x);
        let floor: Vec<u8> = x.iter().map(|v| (v / scale).floor() as i8 as u8).collect();
        let err = check_quantized("floor", &x, &floor, scale).unwrap_err();
        assert!(err.contains("host quantize"), "{err}");
        let rint: Vec<u8> = x.iter().map(|v| (v / scale).round_ties_even() as i8 as u8).collect();
        check_quantized("rint", &x, &rint, scale).unwrap();
        // The scale itself: absmax / 126 is 0.8% off.
        let absmax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(check_quantized("scale", &x, &rint, absmax / 126.0).is_err());
        // A saturating value is clamped to 127, not wrapped.
        let mut clip = rint.clone();
        clip[0] = 128u8;
        assert!(check_quantized("wrap", &x, &clip, scale).is_err());
    }
}
