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

use common::{env_usize, jarr, jnum, jstr, median, ratio_stats, run_paired, Arm, Lcg};
use tessl::nn::{self, I8AScale, I8BLayout, I8BScale, I8Gemm, I8Out};
use tessl::runtime::GpuRuntime;

const MS: [u32; 8] = [1, 16, 64, 128, 256, 512, 1024, 2048];
const N: u32 = 4096;
const K: u32 = 4096;
/// Positions of C checked per M and path: four corners plus seeded draws.
const SAMPLES: usize = 64;
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

    let mut a_bytes = vec![0u8; m_max * K as usize];
    for (i, byte) in a_bytes.iter_mut().enumerate() {
        *byte = i.wrapping_mul(17).wrapping_add(3) as u8;
    }
    a.try_write_bytes(&a_bytes)?;
    let mut b_bytes = vec![0u8; (K as usize) * (N as usize)];
    for (i, byte) in b_bytes.iter_mut().enumerate() {
        *byte = i.wrapping_mul(31).wrapping_add(9) as u8;
    }
    b.try_write_bytes(&b_bytes)?;
    let mut xf = vec![0f32; m_max * K as usize];
    for (i, value) in xf.iter_mut().enumerate() {
        *value = ((i % 251) as f32 - 125.0) * 0.01;
    }
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
    let positions = sample_positions(mu, nu, SEED ^ u64::from(m), SAMPLES);

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
    for &(row, _) in &positions {
        let want = row_scale(&xf[row * ku..(row + 1) * ku]);
        if (f64::from(scales[row]) - f64::from(want)).abs() > 1e-6 * f64::from(want) {
            return Err(format!(
                "quantize_i8_rows M={m} row {row}: scale {} against oracle {want}",
                scales[row]
            ));
        }
    }
    // One host mapping at a time (ft-4b1313de: the runtime lease is exclusive,
    // a second live mapping fails "runtime busy"): copy q's rows out first.
    let q = bufs.q.try_contents_u8()?[..mu * ku].to_vec();
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

/// `absmax / 127`, or 1 for an all-zero row: `quantize_i8_rows`' contract.
fn row_scale(row: &[f32]) -> f32 {
    let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    if absmax == 0.0 {
        1.0
    } else {
        absmax / 127.0
    }
}

/// `count` positions of an `m x n` matrix: its four corners, then seeded draws.
fn sample_positions(m: usize, n: usize, seed: u64, count: usize) -> Vec<(usize, usize)> {
    let mut out = vec![(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)];
    let mut rng = Lcg::new(seed);
    while out.len() < count {
        let r = ((rng.next_u01() * m as f64) as usize).min(m - 1);
        let c = ((rng.next_u01() * n as f64) as usize).min(n - 1);
        out.push((r, c));
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
    use super::{check_sampled, oracle_dot, row_scale, sample_positions};

    #[test]
    fn sample_positions_are_in_range_seeded_and_start_with_the_corners() {
        let p = sample_positions(7, 5, 1, 64);
        assert_eq!(p.len(), 64);
        assert_eq!(&p[..4], &[(0, 0), (0, 4), (6, 0), (6, 4)]);
        assert!(p.iter().all(|&(r, c)| r < 7 && c < 5));
        assert_eq!(p, sample_positions(7, 5, 1, 64));
        assert_ne!(p, sample_positions(7, 5, 2, 64));
        // M = 1 repeats the corners rather than indexing past the matrix.
        assert!(sample_positions(1, 5, 1, 64).iter().all(|&(r, _)| r == 0));
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
}
