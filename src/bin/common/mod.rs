//! Shared benchmark utilities.
//!
//! `bench_gemm_sweep` and `bench_flash_attn` must draw operands from the same
//! RNG and honour the same environment contract, or their numbers are not
//! comparable to each other and a reader has no way to tell. One copy.
//!
//! Not a `src/bin/*.rs` file: cargo would build it as its own binary. A
//! directory without `main.rs` is not a target, which is the same trick
//! `tests/common/mod.rs` uses.

#![allow(dead_code)] // Each binary uses a subset; the module is the union.

/// Operand distribution for the parity dump, selected by `BENCH_PARITY_DIST`.
///
/// One distribution is not a characterisation. Uniform operands are the
/// *easiest* case for a reduced-precision lane: magnitudes are within one
/// order of each other, so every partial sum is well scaled and the K-term
/// accumulation stays far inside its budget. Measured across these five, the
/// fraction of the per-element error budget a lane consumes moves by roughly an
/// order of magnitude — which means a single-distribution number is a floor in
/// the same way a single-shape number was.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dist {
    /// Uniform [-1, 1). The historical default and the benign baseline.
    Uniform,
    /// Standard normal — what activations look like after an RMSNorm.
    Normal,
    /// Sign * 10^U(-3, 3). Six orders of dynamic range inside one dot product,
    /// which is where an 8-bit significand actually hurts.
    LogUniform,
    /// 1 + N(0, 1e-3). Every product is ~1, so the K-term sum is ~K and the
    /// interesting error is in the low bits that cancellation exposes.
    NearCancel,
    /// Clipped Cauchy. A few samples dominate each sum, so the ratio of the
    /// largest term to the total is extreme.
    HeavyTail,
}

impl Dist {
    pub fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "uniform" => Dist::Uniform,
            "normal" => Dist::Normal,
            "log_uniform" => Dist::LogUniform,
            "near_cancel" => Dist::NearCancel,
            "heavy_tail" => Dist::HeavyTail,
            other => {
                return Err(format!(
                    "BENCH_PARITY_DIST={other:?} is not a distribution; expected one of \
                     [uniform, normal, log_uniform, near_cancel, heavy_tail]"
                ))
            }
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Dist::Uniform => "uniform",
            Dist::Normal => "normal",
            Dist::LogUniform => "log_uniform",
            Dist::NearCancel => "near_cancel",
            Dist::HeavyTail => "heavy_tail",
        }
    }
}

/// Deterministic LCG stream in u64. Split out so every distribution draws from
/// the same bit source and a seed means the same thing across all of them.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    /// Next uniform in [0, 1].
    pub fn next_u01(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32) as f64 / (u32::MAX as f64)
    }

    /// Box-Muller, one of the pair. Clamped away from 0 so ln() stays finite.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_u01().max(f64::MIN_POSITIVE);
        let u2 = self.next_u01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Deterministic operand fill. The Python timing lanes use constant 0.5
/// operands instead — GEMM timing is data-independent, so the lanes only share
/// data where it matters: the parity check reads A/B from the .npy files this
/// binary dumps.
///
/// Every value is finite by construction; the dump refuses non-finite operands
/// downstream, and this is where that would otherwise be violated.
pub fn fill_dist(n: usize, seed: u64, dist: Dist) -> Vec<f32> {
    let mut g = Lcg::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let v: f64 = match dist {
            Dist::Uniform => g.next_u01() * 2.0 - 1.0,
            Dist::Normal => g.next_normal(),
            Dist::LogUniform => {
                let sign = if g.next_u01() < 0.5 { -1.0 } else { 1.0 };
                sign * 10f64.powf(g.next_u01() * 6.0 - 3.0)
            }
            Dist::NearCancel => 1.0 + g.next_normal() * 1e-3,
            // Inverse-CDF Cauchy, clipped: the untruncated tail overflows f32
            // and would make the operands themselves non-finite.
            Dist::HeavyTail => {
                let u = g.next_u01().clamp(1e-9, 1.0 - 1e-9);
                (std::f64::consts::PI * (u - 0.5)).tan().clamp(-1e4, 1e4)
            }
        };
        debug_assert!(v.is_finite(), "{} produced a non-finite operand", dist.name());
        out.push(v as f32);
    }
    out
}

/// Fails rather than returning a number nothing sampled. An empty sample set
/// used to index straight off the end of the vector, and `partial_cmp().unwrap()`
/// panicked on a NaN sample instead of naming it.
pub fn median(mut v: Vec<f64>) -> Result<f64, String> {
    if v.is_empty() {
        return Err("median of zero samples (BENCH_ITERS must be >= 1)".to_string());
    }
    if let Some(bad) = v.iter().find(|x| !x.is_finite()) {
        return Err(format!("non-finite timing sample: {bad}"));
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Ok(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Environment integers fail loud. This tree already lost a run to the opposite
/// policy: a malformed `GEMM_FUZZ_SEED` was silently ignored and one seed ran
/// eight times while the log claimed eight. `BENCH_WARMUP` and `BENCH_ITERS`
/// still had that exact `.ok().unwrap_or(default)` shape.
/// This process's lifetime peak physical footprint in bytes: the figure
/// `/usr/bin/time -l` prints as `peak memory footprint`. Not `ru_maxrss`,
/// which under-reads when macOS compresses or reclaims pages
/// (bench/results/qwen35_load_rss_m5pro.txt saw 8.07 GB against 10.79 GB).
pub fn peak_footprint() -> Result<u64, String> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `proc_pid_rusage` fills a `rusage_info_v4` for this flavor, and
    // `info` is one, zero-initialised.
    let rc = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if rc != 0 {
        return Err(format!("proc_pid_rusage: {}", std::io::Error::last_os_error()));
    }
    // SAFETY: zero-initialised and then filled by a successful call.
    Ok(unsafe { info.assume_init() }.ri_lifetime_max_phys_footprint)
}

pub fn env_usize(name: &str, default: usize, min: usize) -> Result<usize, String> {
    let raw = match std::env::var(name) {
        Ok(v) => v,
        Err(std::env::VarError::NotPresent) => return Ok(default),
        Err(e) => return Err(format!("{name}: {e}")),
    };
    parse_usize_env_value(name, &raw, min)
}

fn parse_usize_env_value(name: &str, raw: &str, min: usize) -> Result<usize, String> {
    let v: usize = raw
        .trim()
        .parse()
        .map_err(|_| format!("{name}={raw:?} is not a non-negative integer"))?;
    if v < min {
        return Err(format!("{name}={v} is below the minimum of {min}"));
    }
    Ok(v)
}

/// The measurement-only GEMM kernels live in `kernels/tune/` and are linked
/// only when `TESSL_GEMM_TUNE` is set; the tuning binaries refuse to run
/// without them. The command is `cargo run`, with the variable on it, followed
/// by the binary's name: `TESSL_GEMM_TUNE` is a build-time input, so a plain
/// `cargo run` after a tune `cargo build` reruns build.rs without it and links
/// the metallib without the variants again.
pub const TUNE_RUN: &str = "TESSL_GEMM_TUNE=1 cargo run --release --bin";

/// Exit 2, naming [`TUNE_RUN`] for `bin`, when any of `kernels` has no pipeline in
/// the loaded metallib. Checked before anything is timed: a variant that
/// cannot run must not print a line that reads like a result and let the
/// process exit 0, which is what a page of `skip(pipe)` used to do.
pub fn require_kernels<'a>(rt: &tessl::GpuRuntime, bin: &str, kernels: impl IntoIterator<Item = &'a str>) {
    let mut missing: Vec<&str> = kernels.into_iter().filter(|k| rt.pipeline(k).is_err()).collect();
    missing.sort_unstable();
    missing.dedup();
    if !missing.is_empty() {
        eprintln!(
            "{bin}: {} kernel(s) are not in the loaded metallib: {}\n\
             The tuning variants are opt-in. Rebuild and run with:\n  {TUNE_RUN} {bin}",
            missing.len(),
            missing.join(", ")
        );
        std::process::exit(2);
    }
}

/// Arm `i`'s position in round `round` of an interleaved A/B: forward on
/// even rounds, reversed on odd ones, so every arm runs first and last
/// equally often and a clock that drifts during a round lands on every arm.
pub fn round_order(round: usize, arms: usize) -> Vec<usize> {
    if round % 2 == 0 {
        (0..arms).collect()
    } else {
        (0..arms).rev().collect()
    }
}

/// One arm of an interleaved A/B: a name and one timed call. `run` does
/// `reps` GEMMs and waits for them; a sample is its wall time over `reps`.
pub struct Arm<'a> {
    pub name: String,
    pub reps: usize,
    pub run: Box<dyn FnMut() -> Result<(), String> + 'a>,
}

/// Every arm's per-round median, in ms per GEMM: `rounds[arm][round]`.
pub struct Paired {
    pub names: Vec<String>,
    pub rounds: Vec<Vec<f64>>,
}

/// Time `arms` interleaved: `warmup` untimed calls of each, then `rounds`
/// rounds in [`round_order`], each arm taking `iters` timed calls per round
/// back to back. The comparison is the per-round ratio ([`ratio_stats`]), so
/// drift between rounds cancels; a baseline block and a variant block timed
/// minutes apart (pitfall 1 in docs/benchmarking.md) is what this replaces.
pub fn run_paired(arms: &mut [Arm<'_>], warmup: usize, iters: usize, rounds: usize) -> Result<Paired, String> {
    if iters == 0 || rounds == 0 {
        return Err("an interleaved A/B needs BENCH_ITERS >= 1 and BENCH_ROUNDS >= 1".into());
    }
    for arm in arms.iter_mut() {
        for _ in 0..warmup {
            (arm.run)()?;
        }
    }
    let mut per_round = vec![Vec::with_capacity(rounds); arms.len()];
    for round in 0..rounds {
        for i in round_order(round, arms.len()) {
            let arm = &mut arms[i];
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = std::time::Instant::now();
                (arm.run)()?;
                samples.push(t0.elapsed().as_secs_f64() * 1e3 / arm.reps.max(1) as f64);
            }
            per_round[i].push(median(samples)?);
        }
    }
    Ok(Paired {
        names: arms.iter().map(|a| a.name.clone()).collect(),
        rounds: per_round,
    })
}

/// `base / arm` per round, then the median and range of those ratios
/// (> 1: `arm` is faster). Rounds are paired by index.
pub fn ratio_stats(base: &[f64], arm: &[f64]) -> Result<(f64, f64, f64), String> {
    if base.len() != arm.len() {
        return Err(format!(
            "{} baseline rounds against {} arm rounds",
            base.len(),
            arm.len()
        ));
    }
    let ratios: Vec<f64> = base.iter().zip(arm).map(|(b, a)| b / a).collect();
    let lo = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok((median(ratios)?, lo, hi))
}

/// Largest over smallest of one arm's per-round medians, minus one: how far
/// the arm moved against itself across the run.
pub fn self_spread(rounds: &[f64]) -> f64 {
    let lo = rounds.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = rounds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    hi / lo - 1.0
}

/// The guard docs/benchmarking.md (pitfall 2) quotes: a baseline that moved
/// more than 10% against itself across the rounds of one shape.
pub const SELF_SPREAD_LIMIT: f64 = 0.10;

/// Print the kernels this process dispatched as one `KERNEL_TRACE a,b,...`
/// line on stderr, for `bench/kernel_coverage.py`. Prints nothing when
/// `TESSL_KERNEL_TRACE` is unset, so normal runs are unchanged.
pub fn emit_kernel_trace() {
    if tessl::runtime::kernel_trace_enabled() {
        eprintln!("KERNEL_TRACE {}", tessl::runtime::traced_kernels().join(","));
    }
}

/// Exit 3 when any shape's baseline broke [`SELF_SPREAD_LIMIT`]. Its lines
/// were printed with the `EXCEEDS 10%` flag; the status keeps a script from
/// reading the run as clean.
pub fn exit_if_unstable(bin: &str, unstable: &[String]) {
    if !unstable.is_empty() {
        eprintln!(
            "{bin}: the production baseline moved more than 10% across rounds at {} shape(s): {}; \
             their ratios are not comparable",
            unstable.len(),
            unstable.join(", ")
        );
        std::process::exit(3);
    }
}

#[cfg(test)]
mod tests {
    use super::{median, parse_usize_env_value, ratio_stats, round_order, run_paired, self_spread, Arm};

    #[test]
    fn median_rejects_empty_and_non_finite_samples() {
        assert!(median(Vec::new()).unwrap_err().contains("zero samples"));
        assert!(median(vec![1.0, f64::NAN]).unwrap_err().contains("non-finite"));
        assert!(median(vec![f64::INFINITY]).unwrap_err().contains("non-finite"));
    }

    #[test]
    fn median_handles_odd_and_even_sample_counts() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
    }

    #[test]
    fn round_order_alternates_and_covers_every_arm() {
        assert_eq!(round_order(0, 3), vec![0, 1, 2]);
        assert_eq!(round_order(1, 3), vec![2, 1, 0]);
        assert_eq!(round_order(2, 3), vec![0, 1, 2]);
        assert_eq!(round_order(5, 1), vec![0]);
        assert!(round_order(0, 0).is_empty());
    }

    /// The rounds really interleave: with two arms and three rounds the
    /// calls go A×iters, B×iters, B×iters, A×iters, A×iters, B×iters, after
    /// each arm's warmup. A blocked rig (all of A, then all of B) fails here.
    #[test]
    fn run_paired_interleaves_arms_round_by_round() {
        let log = std::cell::RefCell::new(Vec::new());
        let mut arms = vec![
            Arm {
                name: "a".into(),
                reps: 1,
                run: Box::new(|| {
                    log.borrow_mut().push('a');
                    Ok(())
                }),
            },
            Arm {
                name: "b".into(),
                reps: 1,
                run: Box::new(|| {
                    log.borrow_mut().push('b');
                    Ok(())
                }),
            },
        ];
        let paired = run_paired(&mut arms, 1, 2, 3).unwrap();
        drop(arms);
        // Warmup "ab", then rounds "aabb", "bbaa", "aabb".
        assert_eq!(log.into_inner().into_iter().collect::<String>(), "abaabbbbaaaabb");
        assert_eq!(paired.names, vec!["a", "b"]);
        assert!(paired.rounds.iter().all(|r| r.len() == 3));
        assert!(run_paired(&mut [], 0, 0, 1).is_err());
    }

    #[test]
    fn ratio_stats_pairs_rounds_by_index() {
        let (med, lo, hi) = ratio_stats(&[2.0, 4.0, 3.0], &[1.0, 4.0, 1.0]).unwrap();
        assert_eq!((med, lo, hi), (2.0, 1.0, 3.0));
        assert!(ratio_stats(&[1.0], &[1.0, 2.0]).is_err());
        assert!((self_spread(&[1.0, 1.2, 1.1]) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn benchmark_count_parser_is_strict_and_bounded() {
        assert_eq!(parse_usize_env_value("N", " 7 ", 1).unwrap(), 7);
        assert!(parse_usize_env_value("N", "", 1).is_err());
        assert!(parse_usize_env_value("N", "not-a-count", 1).is_err());
        assert!(parse_usize_env_value("N", "-1", 1).is_err());
        assert!(parse_usize_env_value("N", "0", 1).unwrap_err().contains("minimum"));
    }
}
