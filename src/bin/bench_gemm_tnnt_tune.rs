//! Coop-round A/B for the lanes the NN landing left open: TN, NT, TN-accum
//! bf16 kernels (training backward), and an NN grid swizzle for the
//! large-square operand-reread question. Production baselines go through the
//! public gemm_* API under PrecisionMode::Bf16 with pre-cast operands (the
//! BWD_CAST_ONCE steady state); variants dispatch raw kernels.

// A tuning sweep takes the full GEMM shape plus the tile parameters it is
// sweeping. Bundling them would add a struct that every call site unpacks.
#![allow(clippy::too_many_arguments)]

mod common;

use common::{env_usize, median};
use objc2_metal::MTLComputePipelineState;
use std::time::Instant;
use tessl::gemm::{cast_f32_to_bf16, gemm, gemm_nt_train, gemm_tn_train, GemmBackend};
use tessl::runtime::{mtl_size, GpuRuntime, PrecisionMode};
use tessl::tensor::Tensor;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((((s >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

struct Variant {
    kernel: &'static str,
    sm: usize,
    sn: usize,
    nsg: usize,
    /// Kernel takes tiles_m at buffer(7) (swizzle signature).
    binds_tiles_m: bool,
}

fn dispatch_variant(
    rt: &std::sync::Arc<GpuRuntime>,
    v: &Variant,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    let p = rt.pipeline(v.kernel)?;
    let tiles_n = n / v.sn;
    let tiles_m = m / v.sm;
    let tg = tiles_n * tiles_m;
    let tpt = p.threadExecutionWidth() * v.nsg;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        if v.binds_tiles_m {
            bnd.bind_u32(tiles_m as u32, 7);
        }
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// Round-to-round baseline drift above this makes the ratios uncomparable.
///
/// The blocked predecessor of this binary had no such gate, which is how a run
/// where the *production kernel measured against itself* ranged 0.92x-1.46x
/// could still print a tidy-looking table.
const BASELINE_SPREAD_LIMIT: f64 = 1.10;

/// Every candidate in this rig is expected to be bit-identical to the BF16
/// production path. Keep a small allowance for the split-K production cases,
/// whose changed accumulation order has historically measured around 1e-6.
const MAX_REL_ERR: f64 = 1e-4;

/// The production path for a lane, through the public API.
///
/// Split out so the reference fill and the timed arm dispatch the identical
/// call rather than two hand-copied ones that can drift apart.
fn dispatch_production(lane: Lane, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), String> {
    match lane {
        Lane::Nn => gemm(a, b, c, GemmBackend::TensorOps),
        Lane::Tn => gemm_tn_train(a, b, c, GemmBackend::TensorOps),
        Lane::Nt => gemm_nt_train(a, b, c, GemmBackend::TensorOps),
        // The accum lanes have no public non-flag-gated baseline; callers gate
        // on `has_baseline` rather than reaching here.
        Lane::TnAccum | Lane::NtAccum => Err(String::from(
            "dispatch_production: accum lanes have no API baseline",
        )),
    }
}

/// Ratio max/min across rounds. `INFINITY` if anything is non-positive, so a
/// degenerate run trips the gate rather than reporting a tidy 1.00.
fn spread(v: &[f64]) -> f64 {
    // Checked before the fold, not after: `f64::min` and `f64::max` *ignore* a
    // NaN operand rather than propagating it, so folding first would quietly
    // drop a broken timing and report the spread of whatever remained. A run
    // that produced a NaN has to fail the gate, not pass it with 1.00.
    if v.is_empty() || v.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return f64::INFINITY;
    }
    let lo = v.iter().cloned().fold(f64::MAX, f64::min);
    let hi = v.iter().cloned().fold(0.0, f64::max);
    hi / lo
}

/// Median of the per-round ratios, and the spread of those ratios.
///
/// The median is taken **of the ratios**, not of the two medians. Those are not
/// the same number, and the difference is the whole point of interleaving: each
/// round's pair was measured under one clock state, so `base[r] / var[r]` is a
/// comparison, while `median(base) / median(var)` re-introduces exactly the
/// drift the interleave removed. `median_of_ratios_is_not_ratio_of_medians`
/// pins that.
fn ratio_stats(base: &[f64], var: &[f64]) -> Result<(f64, f64), String> {
    if base.len() != var.len() {
        return Err(format!(
            "ratio series length mismatch: baseline has {} rounds, variant has {}",
            base.len(),
            var.len()
        ));
    }
    if base.is_empty() {
        return Err(String::from("ratio series has zero rounds"));
    }
    if let Some((i, (b, v))) = base
        .iter()
        .zip(var)
        .enumerate()
        .find(|(_, (b, v))| !b.is_finite() || !v.is_finite() || **b <= 0.0 || **v <= 0.0)
    {
        return Err(format!(
            "invalid timing pair at round {i}: baseline={b}, variant={v}"
        ));
    }
    let ratios: Vec<f64> = base.iter().zip(var).map(|(b, v)| b / v).collect();
    let ratio_spread = spread(&ratios);
    Ok((median(ratios)?, ratio_spread))
}

fn time_it_prepared(
    mut prepare: impl FnMut() -> Result<(), String>,
    mut f: impl FnMut() -> Result<(), String>,
    warmup: usize,
    iters: usize,
) -> Result<f64, String> {
    for _ in 0..warmup {
        prepare()?;
        f()?;
    }
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        // Preparation is deliberately outside the interval. Accumulation
        // kernels need C restored before every dispatch, but a host write and
        // its synchronization are not GPU kernel work and must not be billed
        // to the candidate.
        prepare()?;
        let t0 = Instant::now();
        f()?;
        s.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    median(s)
}

fn time_it(
    f: impl FnMut() -> Result<(), String>,
    warmup: usize,
    iters: usize,
) -> Result<f64, String> {
    time_it_prepared(|| Ok(()), f, warmup, iters)
}

fn rel_err(got: &[f32], reference: &[f32]) -> Result<f64, String> {
    if got.len() != reference.len() {
        return Err(format!(
            "correctness length mismatch: got {} values, reference has {}",
            got.len(),
            reference.len()
        ));
    }
    if got.is_empty() {
        return Err(String::from("correctness comparison has zero values"));
    }

    let mut scale = 0.0f64;
    let mut max_abs_err = 0.0f64;
    for (i, (&x, &y)) in got.iter().zip(reference).enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(format!(
                "non-finite correctness value at index {i}: got={x}, reference={y}"
            ));
        }
        scale = scale.max((y as f64).abs());
        max_abs_err = max_abs_err.max((x as f64 - y as f64).abs());
    }
    Ok(max_abs_err / scale.max(1e-12))
}

fn validate_candidate(got: &[f32], reference: &[f32]) -> Result<f64, String> {
    let err = rel_err(got, reference)?;
    if err > MAX_REL_ERR {
        Err(format!(
            "relative error {err:.3e} exceeds {MAX_REL_ERR:.1e}"
        ))
    } else {
        Ok(err)
    }
}

fn require_samples(name: &str, value: usize) -> Result<usize, String> {
    if value == 0 {
        Err(format!("{name} must be >= 1"))
    } else {
        Ok(value)
    }
}

fn require_balanced_rounds(value: usize) -> Result<usize, String> {
    if value < 2 || value % 2 != 0 {
        Err(format!(
            "BENCH_ROUNDS must be an even integer >= 2 for forward/reverse counterbalancing; got {value}"
        ))
    } else {
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimedArm {
    Baseline,
    Candidate(usize),
}

/// Exact positional counterbalance for each adjacent pair of rounds.
///
/// Round 0 runs baseline,c0,c1,... and round 1 runs ...,c1,c0,baseline.
/// Requiring an even round count makes every arm occupy complementary
/// positions equally often, instead of permanently rewarding the first arm.
fn measurement_order(round: usize, has_baseline: bool, candidates: usize) -> Vec<TimedArm> {
    let mut order = Vec::with_capacity(candidates + usize::from(has_baseline));
    if has_baseline {
        order.push(TimedArm::Baseline);
    }
    order.extend((0..candidates).map(TimedArm::Candidate));
    if round % 2 == 1 {
        order.reverse();
    }
    order
}

#[derive(Clone, Copy, PartialEq)]
enum Lane {
    Nn,
    Tn,
    Nt,
    TnAccum,
    NtAccum,
}

fn main() -> Result<(), String> {
    // Parse before creating a GPU runtime: a malformed invocation must fail
    // without touching the device or silently reverting to a different run.
    let warmup = env_usize("BENCH_WARMUP", 8, 0)?;
    let iters = require_samples("BENCH_ITERS", env_usize("BENCH_ITERS", 30, 0)?)?;
    // Interleaved rounds. More rounds is how a ratio earns trust here: the
    // spread across them is reported alongside every number.
    let rounds = require_balanced_rounds(env_usize("BENCH_ROUNDS", 4, 0)?)?;

    let rt = GpuRuntime::new()?;
    rt.set_precision(PrecisionMode::Bf16);

    // (lane, m, n, k, label)
    let cases: &[(Lane, usize, usize, usize, &str)] = &[
        // NN swizzle question (production = landed coop + selection table).
        (Lane::Nn, 4096, 4096, 4096, "nn_square_4096"),
        (Lane::Nn, 2048, 2048, 2048, "nn_square_2048"),
        (Lane::Nn, 8192, 3072, 768, "nn_mlp_up"),
        (Lane::Nn, 4096, 4096, 1024, "nn_tall_k1024"),
        (Lane::Nn, 8192, 768, 3072, "nn_mlp_down"),
        // TN: non-split-K descriptor shapes, then the split-K-gated dW shapes.
        (Lane::Tn, 2048, 2048, 2048, "tn_square_2048"),
        (Lane::Tn, 1024, 1024, 4096, "tn_1024_k4096"),
        (Lane::Tn, 512, 768, 4096, "tn_512x768_k4096"),
        (Lane::Tn, 128, 128, 4096, "tn_dw_attn(splitk)"),
        (Lane::Tn, 128, 384, 4096, "tn_dw_mlp(splitk)"),
        // NT: the dx shapes every backward layer runs, plus generic.
        (Lane::Nt, 4096, 128, 384, "nt_dx_mlp_in"),
        (Lane::Nt, 4096, 384, 128, "nt_dx_mlp_hid"),
        (Lane::Nt, 4096, 128, 128, "nt_dx_attn"),
        (Lane::Nt, 2048, 2048, 2048, "nt_square_2048"),
        (Lane::Nt, 8192, 768, 3072, "nt_wide"),
        // TN accumulate kernels (raw A/B; production kernel is the
        // GEMM_ACCUM=1 path, default-off in training).
        (Lane::TnAccum, 512, 768, 4096, "tnacc_512x768_k4096"),
        (Lane::TnAccum, 2048, 2048, 2048, "tnacc_square_2048"),
        (Lane::NtAccum, 4096, 128, 384, "ntacc_dx_mlp_in"),
        (Lane::NtAccum, 2048, 2048, 2048, "ntacc_square_2048"),
    ];

    let nn_variants = &[
        Variant {
            kernel: "mm_bf16_coop_128x64_sg4_swz4",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
        },
        Variant {
            kernel: "mm_bf16_coop_128x64_sg4_swz8",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
        },
        Variant {
            kernel: "mm_bf16_coop_256x64_sg8_swz4",
            sm: 256,
            sn: 64,
            nsg: 8,
            binds_tiles_m: true,
        },
    ];
    let tn_variants = &[
        Variant {
            kernel: "mm_bf16_tn_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_tn_coop_128x64_sg4",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
    ];
    let nt_variants = &[
        Variant {
            kernel: "mm_bf16_nt_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x64_sg4",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
    ];
    let tnacc_variants = &[
        Variant {
            kernel: "matmul2d_tensorops_tn_accum_bf16_f32",
            sm: 64,
            sn: 32,
            nsg: 4,
            binds_tiles_m: true,
        },
        Variant {
            kernel: "mm_bf16_tn_accum_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
    ];
    let ntacc_variants = &[
        Variant {
            kernel: "matmul2d_tensorops_nt_accum_bf16_f32",
            sm: 64,
            sn: 32,
            nsg: 4,
            binds_tiles_m: true,
        },
        Variant {
            kernel: "mm_bf16_nt_accum_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
    ];

    let mut correctness_rejections = Vec::new();
    for &(lane, m, n, k, label) in cases {
        // Operand storage per lane: NN A[M,K] B[K,N]; TN A[K,M] B[K,N]; NT A[M,K] B[N,K].
        let (a_shape, b_shape) = match lane {
            Lane::Nn => ([m, k], [k, n]),
            Lane::Tn | Lane::TnAccum => ([k, m], [k, n]),
            Lane::Nt | Lane::NtAccum => ([m, k], [n, k]),
        };
        let a_f = rt.alloc_tensor_f32(&a_shape)?;
        let b_f = rt.alloc_tensor_f32(&b_shape)?;
        a_f.buffer.write_f32(&fill(a_shape[0] * a_shape[1], 1));
        b_f.buffer.write_f32(&fill(b_shape[0] * b_shape[1], 2));
        let a = cast_f32_to_bf16(&a_f)?;
        let b = cast_f32_to_bf16(&b_f)?;
        let c_ref = rt.alloc_tensor_f32(&[m, n])?;

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let prefill = 0.25f32;

        // One output buffer per shape, reused by the baseline and every
        // candidate. A fresh allocation per candidate let them pile up until
        // the *baseline* drifted inside a single shape block, which is a
        // measurement artefact indistinguishable from a slow kernel.
        let c = rt.alloc_tensor_f32(&[m, n])?;
        let is_accum = matches!(lane, Lane::TnAccum | Lane::NtAccum);
        let prefill_host = vec![prefill; m * n];

        // Reference, and whether an API baseline exists to ratio against.
        let has_baseline = !is_accum;
        if is_accum {
            // Reference: prefill + product via the matching train path.
            let tmp = rt.alloc_tensor_f32(&[m, n])?;
            if lane == Lane::TnAccum {
                gemm_tn_train(&a, &b, &tmp, GemmBackend::TensorOps)?;
            } else {
                gemm_nt_train(&a, &b, &tmp, GemmBackend::TensorOps)?;
            }
            rt.synchronize()?;
            let base = tmp.buffer.read_f32();
            c_ref
                .buffer
                .write_f32(&base.iter().map(|x| x + prefill).collect::<Vec<_>>());
        } else {
            dispatch_production(lane, &a, &b, &c_ref)?;
            rt.synchronize()?;
        }
        let refv = c_ref.buffer.read_f32()[..m * n].to_vec();

        if has_baseline {
            println!(
                "\n{label}  M={m} N={n} K={k}   ({rounds} interleaved rounds x {iters} iters)"
            );
        } else {
            println!(
                "\n{label}  M={m} N={n} K={k}   (raw accum kernels; ref = TN product + {prefill}; \
                 {rounds} interleaved rounds x {iters} iters)"
            );
        }

        let variants: &[Variant] = match lane {
            Lane::Nn => nn_variants,
            Lane::Tn => tn_variants,
            Lane::Nt => nt_variants,
            Lane::TnAccum => tnacc_variants,
            Lane::NtAccum => ntacc_variants,
        };

        // Establish which candidates are live and correct BEFORE timing, so a
        // skip never costs a round and a wrong kernel is never reported fast.
        let mut live: Vec<(&Variant, f64)> = Vec::new();
        for v in variants {
            if m % v.sm != 0 || n % v.sn != 0 {
                println!("  {:<36}{:>10}", v.kernel, "skip(div)");
                continue;
            }
            if is_accum {
                c.buffer.write_f32(&prefill_host);
            }
            if dispatch_variant(&rt, v, &a, &b, &c, m, n, k).is_err() {
                println!("  {:<36}{:>10}", v.kernel, "skip(pipe)");
                continue;
            }
            rt.synchronize()?;
            let got = c.buffer.read_f32();
            let err = match validate_candidate(&got[..m * n], &refv) {
                Ok(err) => err,
                Err(e) => {
                    let reason = format!("{label}/{}: {e}", v.kernel);
                    println!("  {:<36}{:>10}   {reason}", v.kernel, "REJECT");
                    correctness_rejections.push(reason);
                    continue;
                }
            };
            live.push((v, err));
        }
        if live.is_empty() {
            continue;
        }

        // The interleave. Every arm is timed once per round; adjacent rounds
        // use exact reverse orders. This both keeps clock state paired and
        // prevents an early/late position from permanently favouring one arm.
        let mut base_rounds: Vec<f64> = Vec::with_capacity(rounds);
        let mut var_rounds: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); live.len()];
        for round in 0..rounds {
            for arm in measurement_order(round, has_baseline, live.len()) {
                match arm {
                    TimedArm::Baseline => base_rounds.push(time_it(
                        || {
                            dispatch_production(lane, &a, &b, &c)?;
                            rt.synchronize()
                        },
                        warmup,
                        iters,
                    )?),
                    TimedArm::Candidate(i) => {
                        let v = live[i].0;
                        let ms = if is_accum {
                            time_it_prepared(
                                || {
                                    c.buffer.write_f32(&prefill_host);
                                    Ok(())
                                },
                                || {
                                    dispatch_variant(&rt, v, &a, &b, &c, m, n, k)?;
                                    rt.synchronize()
                                },
                                warmup,
                                iters,
                            )?
                        } else {
                            time_it(
                                || {
                                    dispatch_variant(&rt, v, &a, &b, &c, m, n, k)?;
                                    rt.synchronize()
                                },
                                warmup,
                                iters,
                            )?
                        };
                        var_rounds[i].push(ms);
                    }
                }
            }
        }

        if has_baseline {
            let sp = spread(&base_rounds);
            let med = median(base_rounds.clone())?;
            println!(
                "  {:<36}{:>10.3}{:>12.0}   baseline spread {:.2}",
                "production (API)",
                med,
                flop / (med * 1e6),
                sp
            );
            if sp > BASELINE_SPREAD_LIMIT {
                return Err(format!(
                    "{label}: baseline spread {sp:.3} exceeds {:.0}%; ratios are not comparable",
                    (BASELINE_SPREAD_LIMIT - 1.0) * 100.0
                ));
            }
        }

        println!(
            "  {:<36}{:>10}{:>12}{:>9}{:>9}{:>12}",
            "kernel", "ms(med)", "GFLOP/s", "vs prod", "spread", "rel err"
        );
        for (i, (v, err)) in live.iter().enumerate() {
            let med = median(var_rounds[i].clone())?;
            // Ratio of per-round medians, not a ratio of two blocked medians:
            // each round's pair was measured under the same clock state.
            let (vs, spread_s) = if has_baseline {
                let (med_ratio, sp) = ratio_stats(&base_rounds, &var_rounds[i])?;
                (format!("{med_ratio:>7.2}x"), format!("{sp:>8.2}"))
            } else {
                (String::from("      —"), String::from("       —"))
            };
            println!(
                "  {:<36}{:>10.3}{:>12.0}{}{}{:>12.2e}",
                v.kernel,
                med,
                flop / (med * 1e6),
                vs,
                spread_s,
                err
            );
        }
    }
    if correctness_rejections.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "rejected {} incorrect candidate result(s); rejected candidates were not timed:\n{}",
            correctness_rejections.len(),
            correctness_rejections.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        measurement_order, median, ratio_stats, rel_err, require_balanced_rounds, require_samples,
        spread, time_it, time_it_prepared, validate_candidate, TimedArm, BASELINE_SPREAD_LIMIT,
    };

    #[test]
    fn adjacent_rounds_are_exact_positional_reverses() {
        let forward = measurement_order(0, true, 3);
        let reverse = measurement_order(1, true, 3);
        assert_eq!(
            forward,
            [
                TimedArm::Baseline,
                TimedArm::Candidate(0),
                TimedArm::Candidate(1),
                TimedArm::Candidate(2),
            ]
        );
        assert_eq!(reverse, forward.iter().copied().rev().collect::<Vec<_>>());

        let accum_forward = measurement_order(2, false, 3);
        let accum_reverse = measurement_order(3, false, 3);
        assert_eq!(
            accum_reverse,
            accum_forward.iter().copied().rev().collect::<Vec<_>>()
        );
    }

    #[test]
    fn round_count_must_make_the_counterbalance_complete() {
        assert_eq!(require_balanced_rounds(2).unwrap(), 2);
        assert_eq!(require_balanced_rounds(4).unwrap(), 4);
        for invalid in [0, 1, 3, 5] {
            let error = require_balanced_rounds(invalid).unwrap_err();
            assert!(error.contains("even integer >= 2"), "{error}");
        }
    }

    /// The defect this binary's interleave exists to prevent.
    ///
    /// Blocked timing computes one median for the baseline block and one for
    /// the variant block, then divides. When the clock drifts between the two
    /// blocks that quotient carries the drift. Interleaving pairs each round,
    /// so the drift is common to both terms and cancels.
    ///
    /// Here the machine slows by 3x partway through: rounds 1-2 are fast, 3-4
    /// are throttled. The variant is genuinely 2x the baseline in *every*
    /// round, so the honest answer is 2.00. Ratio-of-medians does not give it.
    #[test]
    fn median_of_ratios_is_not_ratio_of_medians() {
        let base = vec![1.0, 1.0, 3.0, 3.0];
        let var = vec![0.5, 0.5, 1.5, 1.5];

        let (med_ratio, sp) = ratio_stats(&base, &var).unwrap();
        assert!(
            (med_ratio - 2.0).abs() < 1e-12,
            "per-round ratios are all exactly 2.0; got {med_ratio}"
        );
        assert!(
            (sp - 1.0).abs() < 1e-12,
            "every round agrees, so the ratio spread must be 1.0; got {sp}"
        );
    }

    /// The two estimators are not the same statistic.
    ///
    /// When noise lands on the arms unevenly, the baseline's median and the
    /// variant's median can come from rounds that say different things, and
    /// dividing them invents a ratio no round measured. Here the per-round
    /// ratios are 1.0, 0.5 and 2.0 — the variant is a wash — while the ratio of
    /// medians reports it 2x slower.
    ///
    /// The point is not that one number is right: with a 4x spread neither is.
    /// It is that the pairwise form *reports* the 4x, so the gate can refuse
    /// the row, and the blocked form has nowhere to put it.
    #[test]
    fn ratio_of_medians_can_invent_a_ratio_no_round_measured() {
        let base = vec![1.0, 2.0, 10.0];
        let var = vec![1.0, 4.0, 5.0];

        let (med_ratio, sp) = ratio_stats(&base, &var).unwrap();
        let ratio_of_medians = median(base.clone()).unwrap() / median(var.clone()).unwrap();

        assert!((med_ratio - 1.0).abs() < 1e-12, "got {med_ratio}");
        assert!(
            (ratio_of_medians - 0.5).abs() < 1e-12,
            "got {ratio_of_medians}"
        );
        assert!(
            (med_ratio - ratio_of_medians).abs() > 0.1,
            "fixture must separate the estimators"
        );
        assert!(
            sp > BASELINE_SPREAD_LIMIT,
            "a 4x disagreement between rounds must be visible as spread; got {sp}"
        );
    }

    /// A drifting baseline has to trip the gate, because that is the state in
    /// which no ratio on the row means anything.
    #[test]
    fn the_gate_fires_exactly_when_the_baseline_moved() {
        // 0.92x-1.46x is the real observed range of the production kernel
        // measured against itself; it must not pass.
        assert!(spread(&[0.92, 1.46, 1.10]) > BASELINE_SPREAD_LIMIT);
        // A quiet run must not be rejected.
        assert!(spread(&[5.98, 6.02, 6.00]) < BASELINE_SPREAD_LIMIT);
    }

    /// A zero or negative timing is a broken measurement, not a fast one.
    /// Reporting 1.00 for it would read as "quiet machine".
    #[test]
    fn degenerate_timings_fail_the_gate_rather_than_passing_it() {
        assert!(spread(&[0.0, 1.0]).is_infinite());
        assert!(spread(&[-1.0, 1.0]).is_infinite());
        assert!(spread(&[f64::NAN, 1.0]).is_infinite());
        assert!(spread(&[f64::INFINITY, 1.0]).is_infinite());
    }

    #[test]
    fn median_handles_even_and_odd_lengths() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
        assert_eq!(median(vec![7.0]).unwrap(), 7.0);
    }

    #[test]
    fn empty_median_and_zero_bench_iters_fail_loud() {
        let median_err = median(Vec::new()).unwrap_err();
        assert!(median_err.contains("zero samples"), "{median_err}");

        let count_err = require_samples("BENCH_ITERS", 0).unwrap_err();
        assert!(
            count_err.contains("BENCH_ITERS must be >= 1"),
            "{count_err}"
        );

        let timer_err = time_it(|| Ok(()), 0, 0).unwrap_err();
        assert!(
            timer_err.contains("BENCH_ITERS must be >= 1"),
            "{timer_err}"
        );
    }

    #[test]
    fn ratio_stats_rejects_unequal_empty_and_invalid_series() {
        assert!(ratio_stats(&[1.0], &[1.0, 2.0])
            .unwrap_err()
            .contains("length mismatch"));
        assert!(ratio_stats(&[], &[]).unwrap_err().contains("zero rounds"));
        assert!(ratio_stats(&[1.0], &[f64::NAN])
            .unwrap_err()
            .contains("invalid timing pair"));
        assert!(ratio_stats(&[1.0], &[0.0])
            .unwrap_err()
            .contains("invalid timing pair"));
    }

    #[test]
    fn correctness_comparison_rejects_nonfinite_and_length_mismatch() {
        assert!(rel_err(&[1.0], &[1.0, 2.0])
            .unwrap_err()
            .contains("length mismatch"));
        for (got, reference) in [
            (&[f32::NAN][..], &[0.0][..]),
            (&[f32::INFINITY][..], &[0.0][..]),
            (&[0.0][..], &[f32::NAN][..]),
            (&[f32::INFINITY][..], &[f32::INFINITY][..]),
        ] {
            assert!(
                rel_err(got, reference)
                    .unwrap_err()
                    .contains("non-finite correctness value"),
                "got={got:?}, reference={reference:?}"
            );
        }
    }

    #[test]
    fn correctness_threshold_rejects_a_fast_but_wrong_candidate() {
        let err = validate_candidate(&[1.0, 2.0], &[1.0, 1.0]).unwrap_err();
        assert!(err.contains("exceeds"), "{err}");
        assert_eq!(validate_candidate(&[1.0, -2.0], &[1.0, -2.0]).unwrap(), 0.0);
    }

    #[test]
    fn prepared_timer_runs_preparation_before_every_dispatch() {
        use std::cell::RefCell;

        let events = RefCell::new(Vec::new());
        time_it_prepared(
            || {
                events.borrow_mut().push("prepare");
                Ok(())
            },
            || {
                events.borrow_mut().push("dispatch");
                Ok(())
            },
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            events.into_inner(),
            ["prepare", "dispatch", "prepare", "dispatch", "prepare", "dispatch"]
        );
    }
}
