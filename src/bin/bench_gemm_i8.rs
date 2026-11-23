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
//! Before any number is trusted, C is filled with 0xFF bytes (a NaN as f32,
//! f16 and bf16, so a kernel that writes nothing fails) and then sampled
//! positions of C (the four corners, tile edges, one per tile and seeded
//! draws) are compared with a CPU oracle, for both the scalar and the quantized
//! path, as described under "What each check does"; a mismatch or a non-finite
//! value exits 1. The poison, the call and the check run in that order through
//! `checked_call`.
//!
//! Variants (`BENCH_VARIANTS=1`; 0 or unset leaves the default arms, their
//! output and earlier results as they were, and any other value is refused):
//! after the four arms above, every other `nn::gemm_i8` variant is timed at
//! the same shapes with a scalar `a_scale` of 2^-10 (an exact power of two
//! that keeps f16 results finite): B stored `[N, K]` (`TransposedNK`),
//! per-column and per-group (64, 128, 256) `b_scale` on both layouts, `bias`,
//! and the `F16` and `Bf16` stores, alone and combined. They run interleaved
//! (`run_paired`) with two fresh controls, `tessl-i8-gemm` (queues) and
//! `tessl-i8-gemm-rows` (syncs); a variant is compared, per round, with the
//! control that has the same per-call host sync. A `b_scale`, `bias` or
//! row-scale buffer is copied to the host and checked finite
//! (`require_finite`), which waits for every queued command, so every variant
//! with a `b_scale` or `bias` is labelled as syncing on each call and only the
//! rest queue. The bytes copied differ: a per-column variant reads N f32, a
//! group-g variant K/g*N f32 (each group size has its own exactly sized
//! buffer), a bias N f32 and the rows control M f32. The controls match the
//! drain, not the bytes read, so a ratio against them includes that difference.
//! The JSON carries a `variants` object beside `arms`.
//!
//! What each check does (the checks are not all the same strength):
//! - `b_scale` none or per-column, any bias, any store: C must equal, bit for
//!   bit, an f32 replica of the kernel's epilogue in its order (int32 sum to
//!   f32, times `a_scale`, times the column scale, plus the bias, then the
//!   store's cast, round to nearest even). The fused-multiply-add and the
//!   unfused result are both accepted. An int sum off by one fails.
//! - per-group `b_scale`: not replicated exactly, because the kernel's float
//!   accumulation across groups is not modelled. C is compared with an f64
//!   value (the sum over groups of the group's int dot times its scale, times
//!   `a_scale`, plus the bias) inside a window of 1e-6 of the term magnitudes
//!   plus 1e-6 of the result; a 16-bit store must lie within the cast of both
//!   window ends. A wrong layout, scale, group or tile fails; a difference below
//!   the window does not.
//!
//! `BENCH_MS=1,100` replaces the default M list (each entry an unsigned
//! integer without sign, space or leading zero, in 1..=2048), so one locked run
//! can cover M = 1 and M > 64 without the full sweep.

mod common;

use std::collections::HashSet;

use common::{env_usize, jarr, jnum, jstr, median, parse_count, ratio_stats, run_paired, Arm, Lcg};
use tessl::nn::{self, I8AScale, I8BLayout, I8BScale, I8Gemm, I8Out};
use tessl::runtime::GpuRuntime;
use tessl::tensor::{bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits_gpu, f32_to_f16_bits, GpuBuffer};

const MS: [u32; 8] = [1, 16, 64, 128, 256, 512, 1024, 2048];
/// Rows of the operand buffers; the largest M any list may name.
const M_MAX: usize = 2048;
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
    let variants_on = switch("BENCH_VARIANTS", env_usize("BENCH_VARIANTS", 0, 0)?)?;
    let ms = ms_from_env()?;
    let rt = GpuRuntime::new()?;
    let device = rt.device_name();
    let m_max = M_MAX;
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

    let bufs = Bufs {
        a: &a,
        b: &b,
        c: &c,
        x: &x,
        q: &q,
        scales: &scales,
    };
    let variant_bufs = if variants_on {
        Some(VariantBuffers::new(&rt, &b_bytes)?)
    } else {
        None
    };
    let variant_infos = variant_infos();
    let mut variant_rows: Vec<Vec<String>> = vec![Vec::new(); variant_infos.len()];

    let mut rows_by_arm: Vec<Vec<String>> = vec![Vec::new(); ARMS.len()];
    for &m in &ms {
        let pipeline = nn::i8_gemm_pipeline_name(m, I8BLayout::RowMajorKN, 0)?;
        verify_m(&rt, m, pipeline, &bufs, (&a_bytes, &b_bytes, &xf))?;
        if let Some(vb) = &variant_bufs {
            verify_variants(&rt, m, &bufs, vb, (&a_bytes, &b_bytes))?;
        }

        let reps = calls;
        let mut paired_arms = i8_arms(&rt, &bufs, m, reps);
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
        if let Some(vb) = &variant_bufs {
            let paired = time_variants(&rt, m, &bufs, vb, (warmup, iters, rounds, calls))?;
            eprintln!("M={m} variants (a_scale {VARIANT_A_SCALE}, every number follows its oracle check)");
            for (i, (info, per_round)) in variant_infos.iter().zip(&paired.rounds).enumerate() {
                let tail = if i < CONTROLS {
                    String::new()
                } else {
                    let r = reference_arm(info);
                    let (ratio, lo, hi) = ratio_stats(&paired.rounds[r], per_round)?;
                    format!(
                        "  vs {} {ratio:.3} (rounds {lo:.3}..{hi:.3}; > 1 is faster)",
                        variant_infos[r].name
                    )
                };
                eprintln!("  {}{tail}", row(info, m, per_round)?);
                variant_rows[i].push(json_row(info, m, variant_pipeline(m, i)?, per_round)?);
            }
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
    print_arms(&ARMS, &rows_by_arm, "");
    println!("  ]{}", if variants_on { "," } else { "" });
    if variants_on {
        println!("  \"variants\": {{");
        println!("    \"a_scale\": {},", jnum(f64::from(VARIANT_A_SCALE)));
        println!("    \"oracle_checked\": true,");
        println!("    \"arms\": [");
        print_arms(&variant_infos, &variant_rows, "  ");
        println!("    ]");
        println!("  }}");
    }
    println!("}}");
    Ok(())
}

/// The `arms` entries of the JSON, each line indented by `pad` more.
fn print_arms(infos: &[ArmInfo], rows: &[Vec<String>], pad: &str) {
    for (i, info) in infos.iter().enumerate() {
        println!("{pad}    {{");
        println!("{pad}      \"name\": {},", jstr(info.name));
        println!("{pad}      \"what\": {},", jstr(info.what));
        println!(
            "{pad}      \"includes_per_call_host_sync\": {},",
            info.host_sync_per_call
        );
        println!("{pad}      \"rows\": [");
        println!("{pad}        {}", rows[i].join(&format!(",\n{pad}        ")));
        println!("{pad}      ]");
        println!("{pad}    }}{}", if i + 1 < infos.len() { "," } else { "" });
    }
}

/// A 0/1 switch variable's value: anything but 0 or 1 is refused.
fn switch(name: &str, v: usize) -> Result<bool, String> {
    match v {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(format!("{name}={v} is not 0 or 1")),
    }
}

/// `BENCH_MS`: comma-separated M values, each in 1..=[`M_MAX`]; unset is [`MS`].
fn ms_from_env() -> Result<Vec<u32>, String> {
    match std::env::var("BENCH_MS") {
        Ok(raw) => parse_ms(&raw),
        Err(std::env::VarError::NotPresent) => Ok(MS.to_vec()),
        Err(err) => Err(format!("BENCH_MS: {err}")),
    }
}

fn parse_ms(raw: &str) -> Result<Vec<u32>, String> {
    let mut out: Vec<u32> = Vec::new();
    for part in raw.split(',') {
        let m = parse_count("BENCH_MS", part, 1)?;
        if m > M_MAX {
            return Err(format!("BENCH_MS: {m} is outside 1..={M_MAX}"));
        }
        let m = m as u32;
        if out.contains(&m) {
            return Err(format!("BENCH_MS: {m} listed twice"));
        }
        out.push(m);
    }
    Ok(out)
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
    let positions = positions_for(m);

    checked_call(
        || poison_c(bufs.c, mu * nu),
        || {
            call_i8(rt, (bufs.a, bufs.b, bufs.c), m, I8AScale::Scalar(1.0), Tail::plain())?;
            rt.synchronize()
        },
        || {
            let c = bufs.c.try_contents_f32()?;
            check_sampled(
                &format!("gemm_i8 {pipeline} M={m}"),
                HostC::F32(&c[..mu * nu]),
                &HostGemm::plain(&a_bytes[..mu * ku], b_bytes, (nu, ku)),
                &|_| 1.0,
                &positions,
            )
        },
    )?;

    nn::quantize_i8_rows(rt, bufs.x, bufs.q, bufs.scales, m, K)?;
    checked_call(
        || poison_c(bufs.c, mu * nu),
        || {
            call_i8(
                rt,
                (bufs.q, bufs.b, bufs.c),
                m,
                I8AScale::Rows(bufs.scales),
                Tail::plain(),
            )?;
            rt.synchronize()
        },
        || {
            let scales = bufs.scales.try_read_f32()?;
            // One host mapping at a time (ft-4b1313de: the runtime lease is
            // exclusive, a second live mapping fails "runtime busy"): copy q's
            // rows out first.
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
                HostC::F32(&c[..mu * nu]),
                &HostGemm::plain(&q, b_bytes, (nu, ku)),
                &|row| scales[row],
                &positions,
            )
        },
    )
}

/// Fill the first `elems` f32 slots of C with 0xFF bytes: a NaN as f32, as f16
/// and as bf16, so an element a kernel never writes (for any store) is
/// non-finite and fails [`check_sampled`]. `common::poisoned_f32` allocates a
/// fresh buffer, so it cannot re-poison the shared C, and its NaN has a zero
/// low half as 16-bit values.
fn poison_c(c: &GpuBuffer, elems: usize) -> Result<(), String> {
    c.try_write_bytes(&vec![0xFFu8; elems * 4])
}

/// The one order every checked call runs in: poison C, call (and wait), check.
/// A check that could run on a stale C is the failure this order exists to
/// prevent, so no caller sequences the three itself.
fn checked_call(
    poison: impl FnOnce() -> Result<(), String>,
    call: impl FnOnce() -> Result<(), String>,
    check: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    poison()?;
    call()?;
    check()
}

/// The sampled positions of C checked for an `m`-row call.
fn positions_for(m: u32) -> Vec<(usize, usize)> {
    let mu = m as usize;
    sample_positions(mu, N as usize, SEED ^ u64::from(m), SAMPLES, (tile_rows(mu), TILE_N))
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

/// `sum_k a[row,k] * b[k,col]` over signed bytes for `k` in `ks`, exact, and
/// the sum of the magnitudes of the terms (what a float accumulation's error
/// scales with). B is read as the layout says: `[K, N]` or `[N, K]`.
fn oracle_dot_range(
    a: &[u8],
    b: &[u8],
    layout: I8BLayout,
    (n, k): (usize, usize),
    (row, col): (usize, usize),
    ks: std::ops::Range<usize>,
) -> (i64, f64) {
    let (mut sum, mut mag) = (0i64, 0f64);
    for i in ks {
        let bi = match layout {
            I8BLayout::RowMajorKN => i * n + col,
            I8BLayout::TransposedNK => col * k + i,
        };
        let t = i64::from(a[row * k + i] as i8) * i64::from(b[bi] as i8);
        sum += t;
        mag += t.abs() as f64;
    }
    (sum, mag)
}

/// The host copy of a `b_scale` the oracle applies.
#[derive(Clone, Copy)]
enum HostBScale<'a> {
    None,
    /// `N` entries.
    PerColumn(&'a [f32]),
    /// `[K / group, N]`.
    PerGroup {
        scales: &'a [f32],
        group: usize,
    },
}

/// Everything the oracle needs about one GEMM besides C and the A scale.
struct HostGemm<'a> {
    a: &'a [u8],
    b: &'a [u8],
    layout: I8BLayout,
    n: usize,
    k: usize,
    b_scale: HostBScale<'a>,
    bias: Option<&'a [f32]>,
}

impl<'a> HostGemm<'a> {
    /// B `[K, N]`, no `b_scale`, no bias: the default arms.
    fn plain(a: &'a [u8], b: &'a [u8], (n, k): (usize, usize)) -> Self {
        Self {
            a,
            b,
            layout: I8BLayout::RowMajorKN,
            n,
            k,
            b_scale: HostBScale::None,
            bias: None,
        }
    }
}

/// The oracle's value at `(row, col)` in f64 given the row's A scale, and the
/// sum of the magnitudes of its scaled terms (bias included).
fn oracle_value(g: &HostGemm<'_>, row: usize, col: usize, a_scale: f64) -> (f64, f64) {
    let (n, k) = (g.n, g.k);
    let dot = |ks: std::ops::Range<usize>| oracle_dot_range(g.a, g.b, g.layout, (n, k), (row, col), ks);
    let (mut want, mut mag) = match g.b_scale {
        HostBScale::None => {
            let (sum, mag) = dot(0..k);
            (sum as f64 * a_scale, mag * a_scale.abs())
        }
        HostBScale::PerColumn(sc) => {
            let s = f64::from(sc[col]);
            let (sum, mag) = dot(0..k);
            (sum as f64 * a_scale * s, mag * (a_scale * s).abs())
        }
        HostBScale::PerGroup { scales, group } => {
            let (mut want, mut mag) = (0f64, 0f64);
            for gi in 0..k / group {
                let s = f64::from(scales[gi * n + col]);
                let (sum, m) = dot(gi * group..(gi + 1) * group);
                want += sum as f64 * s;
                mag += m * s.abs();
            }
            (want * a_scale, mag * a_scale.abs())
        }
    };
    if let Some(bias) = g.bias {
        let v = f64::from(bias[col]);
        want += v;
        mag += v.abs();
    }
    (want, mag)
}

/// The host view of the C buffer under test: f32, or the raw bits of a 16-bit
/// store.
#[derive(Clone, Copy)]
enum HostC<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    Bf16(&'a [u16]),
}

impl HostC<'_> {
    fn at(&self, i: usize) -> f32 {
        match self {
            Self::F32(c) => c[i],
            Self::F16(c) => f16_bits_to_f32(c[i]),
            Self::Bf16(c) => bf16_bits_to_f32(c[i]),
        }
    }

    /// `x` after the store's documented cast (round to nearest even, f32 for
    /// the f32 store), back as f32.
    fn stored(&self, x: f32) -> f32 {
        match self {
            Self::F32(_) => x,
            Self::F16(_) => f16_bits_to_f32(f32_to_f16_bits(x)),
            Self::Bf16(_) => bf16_bits_to_f32(f32_to_bf16_bits_gpu(x)),
        }
    }
}

/// The values the kernel's epilogue can produce at `(row, col)` when there is
/// no per-group scale, as f32 in the kernel's order: the int32 sum to f32, times
/// `a_scale`, times the column scale if any, plus the bias if any. The last
/// multiply and add may be fused, so the fused and the unfused results are both
/// returned (equal when nothing is added).
fn exact_candidates(g: &HostGemm<'_>, row: usize, col: usize, a_scale: f32) -> Result<[f32; 2], String> {
    let (sum, _) = oracle_dot_range(g.a, g.b, g.layout, (g.n, g.k), (row, col), 0..g.k);
    let s = i32::try_from(sum).map_err(|_| format!("int sum {sum} at [{row},{col}] overflows int32"))? as f32;
    let sa = s * a_scale;
    let colscale = match g.b_scale {
        HostBScale::PerColumn(sc) => Some(sc[col]),
        _ => None,
    };
    let base = colscale.map_or(sa, |bs| sa * bs);
    let bias = g.bias.map(|b| b[col]);
    let unfused = bias.map_or(base, |bi| base + bi);
    let fused = match (colscale, bias) {
        (Some(bs), Some(bi)) => sa.mul_add(bs, bi),
        (None, Some(bi)) => s.mul_add(a_scale, bi),
        _ => unfused,
    };
    Ok([unfused, fused])
}

/// Check `c[row * n + col]` at each position.
///
/// With no per-group scale (`b_scale` None or per-column, any bias, any
/// store) the value must equal, bit for bit, one of [`exact_candidates`]
/// after the store's cast (round to nearest even for f16 and bf16): an int sum
/// off by one, a wrong scale or a missing bias fails.
///
/// A per-group scale is not replicated exactly: the kernel's float
/// accumulation across groups is not modelled. There the value is checked
/// against the f64 oracle ([`oracle_value`]: the sum over groups of the group's
/// int dot times its scale, times the A scale, plus the bias) within 1e-6 of
/// the term magnitudes plus 1e-6 of the result, which an accumulation-order
/// difference stays inside and a misplaced tile or wrong operand does not; a
/// 16-bit store must lie within the cast of both ends of that window, which the
/// monotone cast keeps ordered.
///
/// A non-finite value always fails (an f16 overflow to infinity is documented,
/// so the variants' operands stay finite instead).
fn check_sampled(
    label: &str,
    c: HostC<'_>,
    g: &HostGemm<'_>,
    scale: &dyn Fn(usize) -> f32,
    positions: &[(usize, usize)],
) -> Result<(), String> {
    if let HostBScale::PerGroup { group, .. } = g.b_scale {
        if group == 0 || !g.k.is_multiple_of(group) {
            return Err(format!("{label}: K = {} is not a multiple of group {group}", g.k));
        }
    }
    for &(row, col) in positions {
        let a_scale = scale(row);
        let got32 = c.at(row * g.n + col);
        let got = f64::from(got32);
        let (ok, expected) = match g.b_scale {
            HostBScale::PerGroup { .. } => {
                let (want, mag) = oracle_value(g, row, col, f64::from(a_scale));
                let tol = 1e-6 * mag + 1e-6 * want.abs();
                let in_window = match c {
                    HostC::F32(_) => (got - want).abs() <= tol,
                    _ => {
                        let lo = f64::from(c.stored((want - tol) as f32));
                        let hi = f64::from(c.stored((want + tol) as f32));
                        lo <= got && got <= hi
                    }
                };
                (in_window, format!("{want} (tolerance {tol:e})"))
            }
            _ => {
                let cands = exact_candidates(g, row, col, a_scale)?;
                let hit = cands.iter().any(|&v| c.stored(v).to_bits() == got32.to_bits());
                (hit, format!("{cands:?} (exact)"))
            }
        };
        if !(got32.is_finite() && ok) {
            return Err(format!("{label}: C[{row},{col}] = {got} against oracle {expected}"));
        }
    }
    Ok(())
}

/// The scalar A scale of every variant arm (see the header).
const VARIANT_A_SCALE: f32 = 1.0 / 1024.0;

#[derive(Clone, Copy)]
enum ScaleKind {
    None,
    PerColumn,
    PerGroup(u32),
}

impl ScaleKind {
    /// The `group` argument of `i8_gemm_pipeline_name`.
    fn group(self) -> u32 {
        match self {
            Self::PerGroup(g) => g,
            Self::None | Self::PerColumn => 0,
        }
    }
}

/// One `nn::gemm_i8` variant timed under `BENCH_VARIANTS`.
struct Variant {
    name: &'static str,
    what: &'static str,
    layout: I8BLayout,
    scale: ScaleKind,
    bias: bool,
    out: I8Out,
}

const KN: I8BLayout = I8BLayout::RowMajorKN;
const NT: I8BLayout = I8BLayout::TransposedNK;

const VARIANTS: [Variant; 14] = [
    Variant {
        name: "tessl-i8-gemm-nt",
        what: "nn::gemm_i8 scalar a_scale 2^-10, B stored [N,K] (TransposedNK), f32 store",
        layout: NT,
        scale: ScaleKind::None,
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-percol",
        what: "nn::gemm_i8 scalar a_scale 2^-10, per-column b_scale, B [K,N], f32 store; per-call validation reads N f32 scales",
        layout: KN,
        scale: ScaleKind::PerColumn,
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-percol-nt",
        what: "nn::gemm_i8 scalar a_scale 2^-10, per-column b_scale, B [N,K], f32 store; per-call validation reads N f32 scales",
        layout: NT,
        scale: ScaleKind::PerColumn,
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-g64",
        what: "nn::gemm_i8 scalar a_scale 2^-10, per-group b_scale (group 64), B [K,N], f32 store; per-call validation reads K/64*N f32 scales",
        layout: KN,
        scale: ScaleKind::PerGroup(64),
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-g128",
        what: "nn::gemm_i8 scalar a_scale 2^-10, per-group b_scale (group 128), B [K,N], f32 store; per-call validation reads K/128*N f32 scales",
        layout: KN,
        scale: ScaleKind::PerGroup(128),
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-g256",
        what: "nn::gemm_i8 scalar a_scale 2^-10, per-group b_scale (group 256), B [K,N], f32 store; per-call validation reads K/256*N f32 scales",
        layout: KN,
        scale: ScaleKind::PerGroup(256),
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-g64-nt",
        what: "nn::gemm_i8 scalar a_scale 2^-10, per-group b_scale (group 64), B [N,K], f32 store; per-call validation reads K/64*N f32 scales",
        layout: NT,
        scale: ScaleKind::PerGroup(64),
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-g128-nt",
        what: "nn::gemm_i8 scalar a_scale 2^-10, per-group b_scale (group 128), B [N,K], f32 store; per-call validation reads K/128*N f32 scales",
        layout: NT,
        scale: ScaleKind::PerGroup(128),
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-g256-nt",
        what: "nn::gemm_i8 scalar a_scale 2^-10, per-group b_scale (group 256), B [N,K], f32 store; per-call validation reads K/256*N f32 scales",
        layout: NT,
        scale: ScaleKind::PerGroup(256),
        bias: false,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-bias",
        what: "nn::gemm_i8 scalar a_scale 2^-10, bias, B [K,N], f32 store; per-call validation reads N f32 bias",
        layout: KN,
        scale: ScaleKind::None,
        bias: true,
        out: I8Out::F32,
    },
    Variant {
        name: "tessl-i8-gemm-f16",
        what: "nn::gemm_i8 scalar a_scale 2^-10, B [K,N], f16 store",
        layout: KN,
        scale: ScaleKind::None,
        bias: false,
        out: I8Out::F16,
    },
    Variant {
        name: "tessl-i8-gemm-bf16",
        what: "nn::gemm_i8 scalar a_scale 2^-10, B [K,N], bf16 store",
        layout: KN,
        scale: ScaleKind::None,
        bias: false,
        out: I8Out::Bf16,
    },
    Variant {
        name: "tessl-i8-gemm-nt-g128-bias-f16",
        what: "nn::gemm_i8 scalar a_scale 2^-10, B [N,K], per-group b_scale (group 128), bias, f16 store; per-call validation reads K/128*N f32 scales and N f32 bias",
        layout: NT,
        scale: ScaleKind::PerGroup(128),
        bias: true,
        out: I8Out::F16,
    },
    Variant {
        name: "tessl-i8-gemm-nt-percol-bias-bf16",
        what: "nn::gemm_i8 scalar a_scale 2^-10, B [N,K], per-column b_scale, bias, bf16 store; per-call validation reads N f32 scales and N f32 bias",
        layout: NT,
        scale: ScaleKind::PerColumn,
        bias: true,
        out: I8Out::Bf16,
    },
];

impl Variant {
    /// `gemm_i8` reads a `b_scale` or `bias` buffer on the host to check it is
    /// finite (`require_finite`), which waits for every queued command, so a
    /// variant with either syncs on every call; `None` and no bias queues.
    fn host_sync_per_call(&self) -> bool {
        !matches!(self.scale, ScaleKind::None) || self.bias
    }
}

/// Controls first: `ARMS[0]` (queues, the reference for the variants that queue)
/// and `ARMS[1]` (per-row scale, syncs, the reference for those that sync).
const CONTROLS: usize = 2;

/// The two controls, then every variant: the arms of one paired run.
fn variant_infos() -> Vec<ArmInfo> {
    ARMS[..CONTROLS]
        .iter()
        .map(|a| ArmInfo {
            name: a.name,
            what: a.what,
            host_sync_per_call: a.host_sync_per_call,
            flops: true,
        })
        .chain(VARIANTS.iter().map(|v| ArmInfo {
            name: v.name,
            what: v.what,
            host_sync_per_call: v.host_sync_per_call(),
            flops: true,
        }))
        .collect()
}

/// Index of the control arm a variant arm `i` is compared with: the one with
/// the same per-call host sync.
fn reference_arm(info: &ArmInfo) -> usize {
    usize::from(info.host_sync_per_call)
}

/// Pipeline of arm `i` of [`variant_infos`] at `m`.
fn variant_pipeline(m: u32, i: usize) -> Result<&'static str, String> {
    let (layout, group) = match i.checked_sub(CONTROLS) {
        None => (KN, 0),
        Some(v) => (VARIANTS[v].layout, VARIANTS[v].scale.group()),
    };
    nn::i8_gemm_pipeline_name(m, layout, group)
}

/// Deterministic scales in [0.5, 1.5), distinct per entry.
fn fill_scales(len: usize, seed: u64) -> Vec<f32> {
    (0..len as u64)
        .map(|i| 0.5 + (mix(seed ^ (i << 8)) % 1000) as f32 / 1000.0)
        .collect()
}

/// Bias in [-8, 8], distinct per column.
fn fill_bias(len: usize) -> Vec<f32> {
    (0..len as u64)
        .map(|i| ((mix(0xD ^ (i << 8)) % 16001) as f32 - 8000.0) / 1000.0)
        .collect()
}

/// `b` `[K, N]` as `[N, K]`: the same logical matrix in the other layout.
fn transpose_kn_to_nk(b: &[u8], (n, k): (usize, usize)) -> Vec<u8> {
    let mut out = vec![0u8; b.len()];
    for p in 0..k {
        for j in 0..n {
            out[j * k + p] = b[p * n + j];
        }
    }
    out
}

/// The per-group scale sizes, each with its own exactly sized `[K / group, N]`
/// scale buffer, so a call's validation reads only the bytes it needs.
const GROUPS: [u32; 3] = [64, 128, 256];

/// Host operands of the variants beyond the default arms' A and B.
struct VariantHost {
    b_nk: Vec<u8>,
    col: Vec<f32>,
    /// `[K / group, N]` for each of [`GROUPS`].
    group: Vec<(u32, Vec<f32>)>,
    bias: Vec<f32>,
}

impl VariantHost {
    fn group_scales(&self, group: u32) -> Result<&[f32], String> {
        self.group
            .iter()
            .find(|(g, _)| *g == group)
            .map(|(_, v)| v.as_slice())
            .ok_or_else(|| format!("no scales for group {group}"))
    }
}

/// [`VariantHost`] and its device copies.
struct VariantBuffers {
    host: VariantHost,
    b_nk: GpuBuffer,
    col: GpuBuffer,
    group: Vec<(u32, GpuBuffer)>,
    bias: GpuBuffer,
}

impl VariantBuffers {
    fn new(rt: &std::sync::Arc<GpuRuntime>, b_kn: &[u8]) -> Result<Self, String> {
        let (n, k) = (N as usize, K as usize);
        let host = VariantHost {
            b_nk: transpose_kn_to_nk(b_kn, (n, k)),
            col: fill_scales(n, 0xE),
            group: GROUPS
                .iter()
                .map(|&g| (g, fill_scales(k / g as usize * n, 0xF ^ u64::from(g))))
                .collect(),
            bias: fill_bias(n),
        };
        let b_nk = rt.alloc_buffer(k * n)?;
        b_nk.try_write_bytes(&host.b_nk)?;
        let upload = |v: &[f32]| -> Result<GpuBuffer, String> {
            let buf = rt.alloc_buffer(v.len() * 4)?;
            buf.try_write_f32(v)?;
            Ok(buf)
        };
        let group = host
            .group
            .iter()
            .map(|(g, v)| Ok((*g, upload(v)?)))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            col: upload(&host.col)?,
            bias: upload(&host.bias)?,
            group,
            host,
            b_nk,
        })
    }

    fn group_buf(&self, group: u32) -> Result<&GpuBuffer, String> {
        self.group
            .iter()
            .find(|(g, _)| *g == group)
            .map(|(_, b)| b)
            .ok_or_else(|| format!("no scale buffer for group {group}"))
    }
}

fn call_variant(
    rt: &std::sync::Arc<GpuRuntime>,
    bufs: &Bufs<'_>,
    vb: &VariantBuffers,
    v: &Variant,
    m: u32,
) -> Result<(), String> {
    let b_scale = match v.scale {
        ScaleKind::None => I8BScale::None,
        ScaleKind::PerColumn => I8BScale::PerColumn(&vb.col),
        ScaleKind::PerGroup(group) => I8BScale::PerGroup {
            scales: vb.group_buf(group)?,
            group,
        },
    };
    let b = match v.layout {
        I8BLayout::RowMajorKN => bufs.b,
        I8BLayout::TransposedNK => &vb.b_nk,
    };
    let tail = Tail {
        b_scale,
        bias: v.bias.then_some(&vb.bias),
        layout: v.layout,
        out: v.out,
    };
    call_i8(rt, (bufs.a, b, bufs.c), m, I8AScale::Scalar(VARIANT_A_SCALE), tail)
}

/// Run every variant once at `m` and check sampled positions of C against the
/// oracle, before any variant is timed. C is poisoned before each call.
fn verify_variants(
    rt: &std::sync::Arc<GpuRuntime>,
    m: u32,
    bufs: &Bufs<'_>,
    vb: &VariantBuffers,
    (a_bytes, b_kn): (&[u8], &[u8]),
) -> Result<(), String> {
    let (mu, nu, ku) = (m as usize, N as usize, K as usize);
    let positions = positions_for(m);
    for v in &VARIANTS {
        let gemm = HostGemm {
            a: &a_bytes[..mu * ku],
            b: match v.layout {
                I8BLayout::RowMajorKN => b_kn,
                I8BLayout::TransposedNK => &vb.host.b_nk,
            },
            layout: v.layout,
            n: nu,
            k: ku,
            b_scale: match v.scale {
                ScaleKind::None => HostBScale::None,
                ScaleKind::PerColumn => HostBScale::PerColumn(&vb.host.col),
                ScaleKind::PerGroup(group) => HostBScale::PerGroup {
                    scales: vb.host.group_scales(group)?,
                    group: group as usize,
                },
            },
            bias: v.bias.then_some(&vb.host.bias[..]),
        };
        let label = format!(
            "{} {} M={m}",
            v.name,
            nn::i8_gemm_pipeline_name(m, v.layout, v.scale.group())?
        );
        let scale = |_: usize| VARIANT_A_SCALE;
        checked_call(
            || poison_c(bufs.c, mu * nu),
            || {
                call_variant(rt, bufs, vb, v, m)?;
                rt.synchronize()
            },
            || match v.out {
                I8Out::F32 => {
                    let c = bufs.c.try_contents_f32()?;
                    check_sampled(&label, HostC::F32(&c[..mu * nu]), &gemm, &scale, &positions)
                }
                I8Out::F16 => {
                    let c = bufs.c.try_contents_u16()?;
                    check_sampled(&label, HostC::F16(&c[..mu * nu]), &gemm, &scale, &positions)
                }
                I8Out::Bf16 => {
                    let c = bufs.c.try_contents_u16()?;
                    check_sampled(&label, HostC::Bf16(&c[..mu * nu]), &gemm, &scale, &positions)
                }
            },
        )?;
    }
    Ok(())
}

/// Time the two controls (the first two of [`i8_arms`]) and every variant
/// interleaved at `m`.
fn time_variants(
    rt: &std::sync::Arc<GpuRuntime>,
    m: u32,
    bufs: &Bufs<'_>,
    vb: &VariantBuffers,
    (warmup, iters, rounds, reps): (usize, usize, usize, usize),
) -> Result<common::Paired, String> {
    let mut arms: Vec<Arm<'_>> = i8_arms(rt, bufs, m, reps).into_iter().take(CONTROLS).collect();
    for v in &VARIANTS {
        arms.push(Arm {
            name: v.name.to_string(),
            reps,
            run: Box::new(move || {
                for _ in 0..reps {
                    call_variant(rt, bufs, vb, v, m)?;
                }
                rt.synchronize()
            }),
        });
    }
    run_paired(&mut arms, warmup, iters, rounds)
}

/// What a `gemm_i8` call varies beyond the operands and the activation scale.
struct Tail<'a> {
    b_scale: I8BScale<'a>,
    bias: Option<&'a GpuBuffer>,
    layout: I8BLayout,
    out: I8Out,
}

impl Tail<'_> {
    /// No B scale, no bias, B `[K, N]`, f32 store: the default arms.
    fn plain() -> Tail<'static> {
        Tail {
            b_scale: I8BScale::None,
            bias: None,
            layout: KN,
            out: I8Out::F32,
        }
    }
}

/// The one `nn::gemm_i8` call every arm and check goes through, at this
/// bench's N and K. `(a, b, c)` are the operand and output buffers.
fn call_i8(
    rt: &std::sync::Arc<GpuRuntime>,
    (a, b, c): (&GpuBuffer, &GpuBuffer, &GpuBuffer),
    m: u32,
    a_scale: I8AScale<'_>,
    tail: Tail<'_>,
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
            a_scale,
            b_scale: tail.b_scale,
            bias: tail.bias,
            b_layout: tail.layout,
            out: tail.out,
        },
    )
}

/// The four default arms, in [`ARMS`] order, for `reps` calls per sample at `m`.
fn i8_arms<'a>(rt: &'a std::sync::Arc<GpuRuntime>, bufs: &'a Bufs<'a>, m: u32, reps: usize) -> Vec<Arm<'a>> {
    let gemm = move || call_i8(rt, (bufs.a, bufs.b, bufs.c), m, I8AScale::Scalar(1.0), Tail::plain());
    let rows = move || {
        call_i8(
            rt,
            (bufs.q, bufs.b, bufs.c),
            m,
            I8AScale::Rows(bufs.scales),
            Tail::plain(),
        )
    };
    let quantize = move || nn::quantize_i8_rows(rt, bufs.x, bufs.q, bufs.scales, m, K);
    let arm = |i: usize, call: Box<dyn Fn() -> Result<(), String> + 'a>| Arm {
        name: ARMS[i].name.to_string(),
        reps,
        run: Box::new(move || {
            for _ in 0..reps {
                call()?;
            }
            rt.synchronize()
        }),
    };
    vec![
        arm(0, Box::new(gemm)),
        arm(1, Box::new(rows)),
        arm(2, Box::new(quantize)),
        arm(
            3,
            Box::new(move || {
                quantize()?;
                rows()
            }),
        ),
    ]
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
        amplitude, bf16_bits_to_f32, check_quantized, check_sampled, checked_call, f16_bits_to_f32,
        f32_to_bf16_bits_gpu, f32_to_f16_bits, fill_a, fill_b, fill_bias, fill_scales, fill_x, mix, oracle_dot_range,
        parse_ms, reference_arm, row_scale, sample_positions, switch, tile_rows, transpose_kn_to_nk, variant_infos,
        variant_pipeline, HostBScale, HostC, HostGemm, I8BLayout, I8Out, ScaleKind, CONTROLS, K, KN, MS, M_MAX, N, NT,
        VARIANTS,
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

    /// [`oracle_dot_range`] over all of K.
    fn oracle_dot(a: &[u8], b: &[u8], layout: I8BLayout, (n, k): (usize, usize), row: usize, col: usize) -> (i64, f64) {
        oracle_dot_range(a, b, layout, (n, k), (row, col), 0..k)
    }

    #[test]
    fn oracle_reads_bytes_as_signed_and_b_as_k_by_n() {
        // a = [[1, -1]], b = [[2, 3], [255 (-1), 4]] (K = 2, N = 2).
        let (a, b) = ([1u8, 255], [2u8, 3, 255, 4]);
        assert_eq!(oracle_dot(&a, &b, KN, (2, 2), 0, 0), (3, 3.0));
        assert_eq!(oracle_dot(&a, &b, KN, (2, 2), 0, 1), (-1, 7.0));
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
        check_sampled("t", HostC::F32(&c), &HostGemm::plain(&a, &b, (2, 2)), &scale, &all).unwrap();
        c[3] *= 1.0 + 1e-8;
        check_sampled("t", HostC::F32(&c), &HostGemm::plain(&a, &b, (2, 2)), &scale, &all).unwrap();
    }

    #[test]
    fn check_sampled_rejects_a_wrong_nan_or_misplaced_value() {
        let (a, b, c) = tiny();
        let all = [(0, 0), (0, 1), (1, 0), (1, 1)];
        let scale = |row: usize| if row == 0 { 0.5 } else { 2.0 };
        let mut wrong = c.clone();
        wrong[2] += 1.0;
        let g = HostGemm::plain(&a, &b, (2, 2));
        let err = check_sampled("t", HostC::F32(&wrong), &g, &scale, &all).unwrap_err();
        assert!(err.contains("C[1,0]"), "{err}");
        let mut nan = c.clone();
        nan[0] = f32::NAN;
        assert!(check_sampled("t", HostC::F32(&nan), &g, &scale, &all).is_err());
        // Tiles swapped (the failure a bad tile walk produces): rows exchanged.
        let swapped = vec![c[2], c[3], c[0], c[1]];
        assert!(check_sampled("t", HostC::F32(&swapped), &g, &scale, &all).is_err());
        // A wrong per-row scale is caught, not just a wrong product.
        assert!(check_sampled("t", HostC::F32(&c), &g, &|_| 0.5, &all).is_err());
    }

    /// The corpus the rejection tests share: M = 4, K = 2, N = 4, 2x2 tiles.
    type Grid = (Vec<u8>, Vec<u8>, Vec<f32>, Vec<(usize, usize)>);

    fn grid() -> Grid {
        let a = vec![1u8, 255, 3, 4, 5, 6, 7, 250];
        let b = vec![2u8, 3, 9, 1, 255, 4, 250, 8];
        let scale = |r: usize| 0.5 + r as f32;
        let mut c = Vec::new();
        for r in 0..4 {
            for col in 0..4 {
                let (sum, _) = oracle_dot(&a, &b, KN, (4, 2), r, col);
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
        check_sampled("grid", HostC::F32(c), &HostGemm::plain(a, b, (4, 2)), scale, pos)
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

    /// A logical GEMM the variant tests share: M = 2, N = 3, K = 128, with B
    /// given as a function so both layouts and the references read the same
    /// matrix. `reference` is written from the definitions in f64 and never
    /// calls the oracle under test.
    struct Fixture {
        a: Vec<u8>,
        b_kn: Vec<u8>,
        b_nk: Vec<u8>,
        col: Vec<f32>,
        group: Vec<f32>,
        bias: Vec<f32>,
    }

    const FM: usize = 2;
    const FN: usize = 3;
    const FK: usize = 128;
    const FA: f64 = 0.0625;

    fn logical_b(p: usize, j: usize) -> i8 {
        (((p * 7 + j * 13) % 255) as i32 - 127) as i8
    }

    fn fixture() -> Fixture {
        let a: Vec<u8> = (0..FM * FK).map(|i| (i * 11 % 251) as u8).collect();
        let mut b_kn = vec![0u8; FK * FN];
        for p in 0..FK {
            for j in 0..FN {
                b_kn[p * FN + j] = logical_b(p, j) as u8;
            }
        }
        let b_nk = transpose_kn_to_nk(&b_kn, (FN, FK));
        Fixture {
            a,
            b_kn,
            b_nk,
            col: vec![0.75, 1.25, 0.5],
            group: vec![0.5, 1.5, 1.0, 0.75, 0.25, 2.0],
            bias: vec![1.5, -2.5, 4.0],
        }
    }

    /// f64 reference of C[row][col] from the logical operands.
    fn reference(f: &Fixture, scale: HostBScale<'_>, bias: bool, row: usize, col: usize) -> f64 {
        let dot = |lo: usize, hi: usize| -> f64 {
            (lo..hi)
                .map(|p| f64::from(f.a[row * FK + p] as i8) * f64::from(logical_b(p, col)))
                .sum()
        };
        let v = match scale {
            HostBScale::None => dot(0, FK) * FA,
            HostBScale::PerColumn(sc) => dot(0, FK) * FA * f64::from(sc[col]),
            HostBScale::PerGroup { scales, group } => {
                (0..FK / group)
                    .map(|g| dot(g * group, (g + 1) * group) * f64::from(scales[g * FN + col]))
                    .sum::<f64>()
                    * FA
            }
        };
        v + if bias { f64::from(f.bias[col]) } else { 0.0 }
    }

    fn gemm<'a>(f: &'a Fixture, layout: I8BLayout, scale: HostBScale<'a>, bias: bool) -> HostGemm<'a> {
        HostGemm {
            a: &f.a,
            b: if layout == KN { &f.b_kn } else { &f.b_nk },
            layout,
            n: FN,
            k: FK,
            b_scale: scale,
            bias: bias.then_some(&f.bias[..]),
        }
    }

    fn all_positions() -> Vec<(usize, usize)> {
        (0..FM).flat_map(|r| (0..FN).map(move |c| (r, c))).collect()
    }

    /// C as a kernel would store it in f32. A per-group scale takes the f64
    /// reference; otherwise the f32 epilogue in the kernel's order, computed
    /// here from the logical operands (every value is exact in f32).
    fn c_of(f: &Fixture, scale: HostBScale<'_>, bias: bool) -> Vec<f32> {
        all_positions()
            .iter()
            .map(|&(r, c)| {
                if matches!(scale, HostBScale::PerGroup { .. }) {
                    return reference(f, scale, bias, r, c) as f32;
                }
                let sum: i32 = (0..FK)
                    .map(|p| i32::from(f.a[r * FK + p] as i8) * i32::from(logical_b(p, c)))
                    .sum();
                let sa = sum as f32 * FA as f32;
                let base = match scale {
                    HostBScale::PerColumn(sc) => sa * sc[c],
                    _ => sa,
                };
                if bias {
                    base + f.bias[c]
                } else {
                    base
                }
            })
            .collect()
    }

    /// One int unit in the sum moves an f32 result by a_scale (times the column
    /// scale): the exact check rejects it, in both layouts, with and without a
    /// column scale and bias.
    #[test]
    fn the_exact_check_rejects_an_int_sum_off_by_one() {
        let f = fixture();
        for (scale, bias) in [
            (HostBScale::None, false),
            (HostBScale::None, true),
            (HostBScale::PerColumn(&f.col), false),
            (HostBScale::PerColumn(&f.col), true),
        ] {
            let c = c_of(&f, scale, bias);
            for layout in [KN, NT] {
                check(HostC::F32(&c), &gemm(&f, layout, scale, bias)).unwrap();
                for (i, dir) in [(0usize, 1.0f32), (3, -1.0)] {
                    let unit = match scale {
                        HostBScale::PerColumn(sc) => sc[i % FN],
                        _ => 1.0,
                    } * FA as f32;
                    let mut off = c.clone();
                    off[i] += dir * unit;
                    assert!(check(HostC::F32(&off), &gemm(&f, layout, scale, bias)).is_err());
                }
            }
        }
    }

    /// Where the multiply-add fuses, its result differs from the unfused one:
    /// both are accepted and a value one ulp from either is not.
    #[test]
    fn the_exact_check_accepts_the_fused_and_the_unfused_result_only() {
        let (a, b) = ([1u8], [1u8]);
        let (a_scale, col, bias) = (f32::from_bits(0x3F80_0001), [f32::from_bits(0x3F80_0002)], [-1.0f32]);
        let g = HostGemm {
            a: &a,
            b: &b,
            layout: KN,
            n: 1,
            k: 1,
            b_scale: HostBScale::PerColumn(&col),
            bias: Some(&bias),
        };
        let sa = 1.0f32 * a_scale;
        let unfused = sa * col[0] + bias[0];
        let fused = sa.mul_add(col[0], bias[0]);
        assert_ne!(unfused.to_bits(), fused.to_bits());
        let run = |v: f32| check_sampled("fx", HostC::F32(&[v]), &g, &|_| a_scale, &[(0, 0)]);
        run(unfused).unwrap();
        run(fused).unwrap();
        // The two results are adjacent floats: the first value outside them on
        // either side is rejected.
        let (lo, hi) = (
            unfused.to_bits().min(fused.to_bits()),
            unfused.to_bits().max(fused.to_bits()),
        );
        assert!(run(f32::from_bits(lo - 1)).is_err());
        assert!(run(f32::from_bits(hi + 1)).is_err());
    }

    /// A stale C that already holds the expected values passes the check, so
    /// the order must poison it first: with the poison in place a kernel that
    /// writes nothing is rejected as non-finite, in every store; with the
    /// poison skipped the stale values would pass.
    #[test]
    fn checked_call_poisons_c_so_a_kernel_that_writes_nothing_fails() {
        let f = fixture();
        let scale = HostBScale::PerColumn(&f.col);
        let want = c_of(&f, scale, true);
        let g = gemm(&f, NT, scale, true);
        let h: Vec<u16> = want.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let b: Vec<u16> = want.iter().map(|&v| f32_to_bf16_bits_gpu(v)).collect();
        // The stale buffers are accepted on their own.
        check(HostC::F32(&want), &g).unwrap();
        check(HostC::F16(&h), &g).unwrap();
        check(HostC::Bf16(&b), &g).unwrap();
        for kind in 0..3 {
            let c32 = std::cell::RefCell::new(want.clone());
            let c16 = std::cell::RefCell::new(if kind == 1 { h.clone() } else { b.clone() });
            let called = std::cell::Cell::new(false);
            let res = checked_call(
                || {
                    // The real poison: 0xFF bytes over the buffer.
                    let bytes = vec![0xFFu8; want.len() * 4];
                    let words: Vec<f32> = bytes
                        .chunks(4)
                        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                        .collect();
                    *c32.borrow_mut() = words;
                    let halves: Vec<u16> = bytes
                        .chunks(2)
                        .take(want.len())
                        .map(|w| u16::from_le_bytes([w[0], w[1]]))
                        .collect();
                    *c16.borrow_mut() = halves;
                    Ok(())
                },
                || {
                    called.set(true); // a kernel that writes nothing
                    Ok(())
                },
                || match kind {
                    0 => check(HostC::F32(&c32.borrow()), &g),
                    1 => check(HostC::F16(&c16.borrow()), &g),
                    _ => check(HostC::Bf16(&c16.borrow()), &g),
                },
            );
            assert!(called.get());
            assert!(res.is_err(), "store {kind} passed on an unwritten C");
        }
        // The order: poison, call, check.
        let order = std::cell::RefCell::new(Vec::new());
        checked_call(
            || {
                order.borrow_mut().push("poison");
                Ok(())
            },
            || {
                order.borrow_mut().push("call");
                Ok(())
            },
            || {
                order.borrow_mut().push("check");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*order.borrow(), ["poison", "call", "check"]);
        // A failing poison stops the call.
        let ran = std::cell::Cell::new(false);
        assert!(checked_call(
            || Err("x".into()),
            || {
                ran.set(true);
                Ok(())
            },
            || Ok(())
        )
        .is_err());
        assert!(!ran.get());
    }

    #[test]
    fn bench_switches_accept_only_zero_and_one() {
        assert!(!switch("BENCH_VARIANTS", 0).unwrap());
        assert!(switch("BENCH_VARIANTS", 1).unwrap());
        for bad in [2usize, 3, 10, usize::MAX] {
            assert!(switch("BENCH_VARIANTS", bad).is_err());
        }
    }

    fn wrap(f16: bool, bits: &[u16]) -> HostC<'_> {
        if f16 {
            HostC::F16(bits)
        } else {
            HostC::Bf16(bits)
        }
    }

    fn check(c: HostC<'_>, g: &HostGemm<'_>) -> Result<(), String> {
        check_sampled("fx", c, g, &|_| FA as f32, &all_positions())
    }

    #[test]
    fn transpose_moves_each_element_and_round_trips() {
        let kn = [1u8, 2, 3, 4, 5, 6]; // K = 2, N = 3
        let nk = transpose_kn_to_nk(&kn, (3, 2));
        assert_eq!(nk, [1, 4, 2, 5, 3, 6]);
        assert_eq!(transpose_kn_to_nk(&nk, (2, 3)), kn);
    }

    #[test]
    fn oracle_reads_b_in_the_layout_it_is_told() {
        let f = fixture();
        for (row, col) in all_positions() {
            assert_eq!(
                oracle_dot(&f.a, &f.b_kn, KN, (FN, FK), row, col),
                oracle_dot(&f.a, &f.b_nk, NT, (FN, FK), row, col)
            );
        }
        // The same bytes read in the other layout are a different matrix.
        let differs = all_positions().iter().any(|&(r, c)| {
            oracle_dot(&f.a, &f.b_nk, KN, (FN, FK), r, c) != oracle_dot(&f.a, &f.b_nk, NT, (FN, FK), r, c)
        });
        assert!(differs);
    }

    /// C from a kernel that read B as `[N, K]` is accepted under `NT` and
    /// rejected when the oracle reads the same bytes as `[K, N]`.
    #[test]
    fn the_oracle_rejects_a_wrong_layout_read() {
        let f = fixture();
        let c = c_of(&f, HostBScale::None, false);
        check(HostC::F32(&c), &gemm(&f, KN, HostBScale::None, false)).unwrap();
        check(HostC::F32(&c), &gemm(&f, NT, HostBScale::None, false)).unwrap();
        let mut wrong = gemm(&f, NT, HostBScale::None, false);
        wrong.layout = KN;
        assert!(check(HostC::F32(&c), &wrong).is_err());
        let mut wrong = gemm(&f, KN, HostBScale::None, false);
        wrong.b = &f.b_nk;
        assert!(check(HostC::F32(&c), &wrong).is_err());
    }

    #[test]
    fn the_oracle_applies_and_rejects_the_per_column_scale() {
        let f = fixture();
        let scale = HostBScale::PerColumn(&f.col);
        let c = c_of(&f, scale, false);
        for layout in [KN, NT] {
            check(HostC::F32(&c), &gemm(&f, layout, scale, false)).unwrap();
            // The scale ignored, or read for the neighbouring column.
            assert!(check(HostC::F32(&c), &gemm(&f, layout, HostBScale::None, false)).is_err());
            let swapped = [f.col[1], f.col[0], f.col[2]];
            assert!(check(
                HostC::F32(&c),
                &gemm(&f, layout, HostBScale::PerColumn(&swapped), false)
            )
            .is_err());
        }
    }

    #[test]
    fn the_oracle_flushes_per_group_and_rejects_a_collapsed_or_wrong_group() {
        let f = fixture();
        for group in [64usize, 128] {
            let scale = HostBScale::PerGroup {
                scales: &f.group,
                group,
            };
            let c = c_of(&f, scale, false);
            for layout in [KN, NT] {
                check(HostC::F32(&c), &gemm(&f, layout, scale, false)).unwrap();
            }
        }
        // Two groups with different scales: C built as if group 0's scale
        // covered all of K (a collapsed flush) is rejected, and so is group 64
        // read as 128.
        let g64 = HostBScale::PerGroup {
            scales: &f.group,
            group: 64,
        };
        let g128 = HostBScale::PerGroup {
            scales: &f.group,
            group: 128,
        };
        let c64 = c_of(&f, g64, false);
        assert!(check(HostC::F32(&c64), &gemm(&f, KN, g128, false)).is_err());
        let collapsed = c_of(&f, g128, false);
        assert!(check(HostC::F32(&collapsed), &gemm(&f, KN, g64, false)).is_err());
        // K not a multiple of the group is refused, not truncated.
        let g96 = HostBScale::PerGroup {
            scales: &f.group,
            group: 96,
        };
        let err = check(HostC::F32(&c64), &gemm(&f, KN, g96, false)).unwrap_err();
        assert!(err.contains("multiple of group"), "{err}");
    }

    #[test]
    fn the_oracle_adds_and_rejects_a_missing_or_misplaced_bias() {
        let f = fixture();
        let c = c_of(&f, HostBScale::None, true);
        check(HostC::F32(&c), &gemm(&f, KN, HostBScale::None, true)).unwrap();
        assert!(check(HostC::F32(&c), &gemm(&f, KN, HostBScale::None, false)).is_err());
        let rotated = [f.bias[1], f.bias[2], f.bias[0]];
        let mut g = gemm(&f, KN, HostBScale::None, false);
        g.bias = Some(&rotated);
        assert!(check(HostC::F32(&c), &g).is_err());
    }

    /// The 16-bit stores are checked against the documented cast: the cast
    /// value passes, a neighbouring half or bf16 value fails, and so do an
    /// uncast f32, an infinity and the bits read as the other format.
    #[test]
    fn the_oracle_checks_16_bit_stores_against_the_cast() {
        let f = fixture();
        let scale = HostBScale::PerColumn(&f.col);
        let want = c_of(&f, scale, true);
        let g = gemm(&f, KN, scale, true);
        let h: Vec<u16> = want.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let b: Vec<u16> = want.iter().map(|&v| f32_to_bf16_bits_gpu(v)).collect();
        check(HostC::F16(&h), &g).unwrap();
        check(HostC::Bf16(&b), &g).unwrap();
        for (bits, f16) in [(&h, true), (&b, false)] {
            let mut next = bits.clone();
            next[3] += 1;
            assert!(check(wrap(f16, &next), &g).is_err());
            let mut prev = bits.clone();
            prev[0] -= 1;
            assert!(check(wrap(f16, &prev), &g).is_err());
        }
        assert!(check(HostC::Bf16(&h), &g).is_err());
        assert!(check(HostC::F16(&b), &g).is_err());
        let mut inf = h.clone();
        inf[1] = f32_to_f16_bits(f32::INFINITY);
        assert!(check(HostC::F16(&inf), &g).is_err());
        // The f32 oracle's value is not a 16-bit value: the cast must be applied.
        assert!(want.iter().any(|&v| f16_bits_to_f32(f32_to_f16_bits(v)) != v));
        assert!(want.iter().any(|&v| bf16_bits_to_f32(f32_to_bf16_bits_gpu(v)) != v));
    }

    #[test]
    fn variant_operands_are_distinct_and_in_range() {
        let s = fill_scales(4096, 7);
        assert!(s.iter().all(|v| (0.5..1.5).contains(v)));
        assert!(s.windows(2).any(|w| w[0] != w[1]));
        let bias = fill_bias(4096);
        assert!(bias.iter().all(|v| v.abs() <= 8.0) && bias.windows(2).any(|w| w[0] != w[1]));
        assert_ne!(mix(1), mix(2));
    }

    /// Every variant the lane exists to cover is in the table, names are unique
    /// and every pipeline name resolves (so a group that does not divide K, or
    /// an invalid group, fails here and not on the device).
    #[test]
    fn the_variant_table_covers_every_gemm_i8_variant() {
        let names: std::collections::HashSet<_> = VARIANTS.iter().map(|v| v.name).collect();
        assert_eq!(names.len(), VARIANTS.len());
        for layout in [KN, NT] {
            for group in [64u32, 128, 256] {
                assert!(
                    VARIANTS
                        .iter()
                        .any(|v| v.layout == layout && matches!(v.scale, ScaleKind::PerGroup(g) if g == group)),
                    "{layout:?} group {group}"
                );
            }
            assert!(VARIANTS
                .iter()
                .any(|v| v.layout == layout && matches!(v.scale, ScaleKind::PerColumn)));
        }
        assert!(VARIANTS.iter().any(|v| v.bias));
        assert!(VARIANTS.iter().any(|v| v.out == I8Out::F16));
        assert!(VARIANTS.iter().any(|v| v.out == I8Out::Bf16));
        for v in &VARIANTS {
            if let ScaleKind::PerGroup(g) = v.scale {
                assert_eq!(K % g, 0, "{}", v.name);
            }
        }
        for m in [1u32, 2, 64, 65, 2048] {
            for i in 0..VARIANTS.len() + CONTROLS {
                variant_pipeline(m, i).unwrap();
            }
        }
        // Arm 2 is the first variant (NT, no scale), arm 3 the second (per-column, KN).
        assert_eq!(variant_pipeline(65, 2).unwrap(), "i8_w8a8_128_nt");
        assert_eq!(variant_pipeline(65, 3).unwrap(), "i8_w8a8_128");
        assert_eq!(variant_pipeline(1, 2).unwrap(), "i8_gemv_dequant");
        assert!(variant_pipeline(0, 2).is_err());
    }

    /// The label follows what the library does: a b_scale or bias buffer is
    /// read on the host, so those arms sync; the plain ones queue. Each arm is
    /// compared with the control that has the same sync.
    #[test]
    fn host_sync_labels_follow_b_scale_and_bias() {
        for v in &VARIANTS {
            let expect = !matches!(v.scale, ScaleKind::None) || v.bias;
            assert_eq!(v.host_sync_per_call(), expect, "{}", v.name);
        }
        let by_name = |n: &str| VARIANTS.iter().find(|v| v.name == n).unwrap().host_sync_per_call();
        assert!(!by_name("tessl-i8-gemm-nt"));
        assert!(!by_name("tessl-i8-gemm-f16"));
        assert!(by_name("tessl-i8-gemm-percol"));
        assert!(by_name("tessl-i8-gemm-g128-nt"));
        assert!(by_name("tessl-i8-gemm-bias"));
        let infos = variant_infos();
        assert_eq!(infos.len(), VARIANTS.len() + CONTROLS);
        assert!(!infos[0].host_sync_per_call && infos[1].host_sync_per_call);
        for info in &infos[CONTROLS..] {
            assert_eq!(infos[reference_arm(info)].host_sync_per_call, info.host_sync_per_call);
        }
    }

    #[test]
    fn bench_ms_parses_a_list_and_rejects_bad_entries() {
        assert_eq!(parse_ms("1,100").unwrap(), vec![1, 100]);
        assert_eq!(parse_ms("2048").unwrap(), vec![2048]);
        for bad in [
            "", "0", "2049", "x", "1,,2", "1,1", "-3", "+5", "01", " 1", "1 ", "1, 100",
        ] {
            assert!(parse_ms(bad).is_err(), "{bad:?}");
        }
        assert!(MS.iter().all(|&m| m as usize <= M_MAX));
    }
}
