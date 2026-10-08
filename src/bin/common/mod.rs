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
        debug_assert!(
            v.is_finite(),
            "{} produced a non-finite operand",
            dist.name()
        );
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

#[cfg(test)]
mod tests {
    use super::{median, parse_usize_env_value};

    #[test]
    fn median_rejects_empty_and_non_finite_samples() {
        assert!(median(Vec::new()).unwrap_err().contains("zero samples"));
        assert!(median(vec![1.0, f64::NAN])
            .unwrap_err()
            .contains("non-finite"));
        assert!(median(vec![f64::INFINITY])
            .unwrap_err()
            .contains("non-finite"));
    }

    #[test]
    fn median_handles_odd_and_even_sample_counts() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
    }

    #[test]
    fn benchmark_count_parser_is_strict_and_bounded() {
        assert_eq!(parse_usize_env_value("N", " 7 ", 1).unwrap(), 7);
        assert!(parse_usize_env_value("N", "", 1).is_err());
        assert!(parse_usize_env_value("N", "not-a-count", 1).is_err());
        assert!(parse_usize_env_value("N", "-1", 1).is_err());
        assert!(parse_usize_env_value("N", "0", 1)
            .unwrap_err()
            .contains("minimum"));
    }
}
