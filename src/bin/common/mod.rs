//! Shared benchmark utilities.
//!
//! One owner for what every benchmark binary must agree on, or their numbers
//! are not comparable and a reader has no way to tell: the operand RNG
//! ([`fill`], [`fill_dist`]), the median ([`median`], which refuses an empty or
//! non-finite sample set), the environment contract ([`env_usize`], which
//! refuses a malformed or below-minimum count), and the interleaved A/B
//! statistics ([`run_paired`], [`ratio_stats`], [`self_spread`]), and the
//! profile binaries' quiet-machine gate ([`gated`], [`others_running`]) with
//! their JSON scalar writers ([`jnum`], [`jarr`]). A binary
//! that needs one adds it here; it does not carry a copy.
//!
//! Declared with `mod common;` by (`rg -l '^mod common;' src/bin`):
//! bench_embedgemma2, bench_embedgemma2_mm, bench_embedgemma2_vision, bench_encoder_attn, bench_flash_attn, bench_gemm_coop_tile,
//! bench_gemm_routes, bench_gemm_sweep, bench_gemm_tile_tune,
//! bench_gemm_tnnt_tune, bench_gemm_variants, bench_nn_kernels, bench_paired,
//! bench_qwen35_layers, bench_qwen35_train, probe_gdn_scan, probe_gemm_bits,
//! probe_gemm_parity, probe_load_memory, profile_embedgemma2, bench_embedgemma2_audio,
//! profile_embedgemma2_audio, profile_embedgemma2_vision (profile_embedgemma2 still
//! carries its own copy of the gate and the scalar writers, which this module took
//! from it).
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

/// The uniform [-1, 1) operand fill every bench draws from: `fill_dist` with
/// [`Dist::Uniform`], the same LCG stream the per-binary copies used.
pub fn fill(n: usize, seed: u64) -> Vec<f32> {
    fill_dist(n, seed, Dist::Uniform)
}

/// [`fill`] mapped through `x * scale + shift`, for operands that are not
/// centred on zero or are smaller than one.
pub fn fill_affine(n: usize, seed: u64, scale: f32, shift: f32) -> Vec<f32> {
    let mut v = fill(n, seed);
    for x in &mut v {
        *x = *x * scale + shift;
    }
    v
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

// The peak-footprint reader lives in its own file so the integration tests
// that print their peak include the same source.
#[path = "footprint.rs"]
mod footprint;
#[allow(unused_imports)] // Each bin uses a subset of `common`.
pub use footprint::peak_footprint;

/// The largest count any benchmark knob (`BENCH_ITERS`, `--rounds`, ...) may
/// name. No bench runs a million of anything; the bound turns `2^64 - 1`, and a
/// typo that adds digits, into an error instead of a run that never ends.
pub const MAX_COUNT: usize = 999_999;

/// Environment integers fail loud. This tree already lost a run to the opposite
/// policy: a malformed `GEMM_FUZZ_SEED` was silently ignored and one seed ran
/// eight times while the log claimed eight. (See [`env_usize`].)
///
/// The one strict count parser: ASCII digits only (no sign, no surrounding
/// whitespace), no leading zero unless the whole value is `0`, between `min`
/// and [`MAX_COUNT`]. `name` is the variable or flag, for the message.
pub fn parse_count(name: &str, raw: &str, min: usize) -> Result<usize, String> {
    let digits = !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()) && (raw == "0" || !raw.starts_with('0'));
    let v = digits.then(|| raw.parse::<usize>().ok()).flatten().ok_or_else(|| {
        format!("{name}={raw:?} is not a non-negative integer (ASCII digits, no sign or leading zero)")
    })?;
    if v < min {
        return Err(format!("{name}={v} is below the minimum of {min}"));
    }
    if v > MAX_COUNT {
        return Err(format!("{name}={v} is above the maximum of {MAX_COUNT}"));
    }
    Ok(v)
}

/// `name` from the environment through [`parse_count`], `default` when unset.
pub fn env_usize(name: &str, default: usize, min: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(v) => parse_count(name, &v, min),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(format!("{name}: {e}")),
    }
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

/// A process-unique suffix for a staged file name.
pub fn unique_stamp() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{:x}-{nanos:x}", std::process::id())
}

/// Refuse an existing `path` that is not a regular file, without blocking.
///
/// The open is `O_NONBLOCK` and the answer comes from `fstat` of that handle,
/// as `tessl::plain::open_regular_file` does for checkpoints: a FIFO with no
/// reader or writer must not hang the caller, and a device or directory is not
/// something to replace. A missing path is fine.
pub fn refuse_non_regular_target(path: &std::path::Path) -> Result<(), String> {
    use std::os::unix::fs::OpenOptionsExt;
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
    {
        Ok(f) => {
            let md = f.metadata().map_err(|e| format!("{}: {e}", path.display()))?;
            if md.is_file() {
                Ok(())
            } else {
                Err(format!(
                    "{}: exists and is not a regular file; refusing to replace it",
                    path.display()
                ))
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// Replace `path` with `bytes` durably, exposing no partial content: a staged
/// file in the same directory is synced, renamed over `path`, and the
/// directory synced. A non-regular existing target is refused first.
pub fn atomic_write(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    let abs = std::path::absolute(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = abs
        .parent()
        .ok_or_else(|| format!("{}: no parent directory", abs.display()))?;
    let name = abs.file_name().and_then(|n| n.to_str()).unwrap_or("out");
    if !dir.is_dir() {
        return Err(format!(
            "{}: parent directory does not exist (never created implicitly)",
            abs.display()
        ));
    }
    refuse_non_regular_target(&abs)?;
    let staged = dir.join(format!(".{name}.{}", unique_stamp()));
    let result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(&staged)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&staged, &abs)?;
        std::fs::File::open(dir)?.sync_all()
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&staged);
        return Err(format!("{}: {e}", abs.display()));
    }
    Ok(())
}

/// Print the kernels this process dispatched as one `KERNEL_TRACE a,b,...`
/// line on stderr, for the `kernel_coverage` bin. Prints nothing when
/// `TESSL_KERNEL_TRACE` is unset, so normal runs are unchanged.
pub fn emit_kernel_trace() {
    if tessl::runtime::kernel_trace_enabled() {
        eprintln!("KERNEL_TRACE {}", tessl::runtime::traced_kernels().join(","));
    }
}

// ---------------------------------------------------------------------------
// Kernel lanes: one public entry point on small random operands
// ---------------------------------------------------------------------------

/// A device buffer holding `data` (f32).
pub fn upload_f32(rt: &std::sync::Arc<tessl::GpuRuntime>, data: &[f32]) -> Result<tessl::tensor::GpuBuffer, String> {
    let b = rt.alloc_buffer(data.len().max(1) * 4)?;
    b.write_f32(data);
    Ok(b)
}

/// A device buffer holding `data` as bf16 (round to nearest even).
pub fn upload_bf16(rt: &std::sync::Arc<tessl::GpuRuntime>, data: &[f32]) -> Result<tessl::tensor::GpuBuffer, String> {
    let b = rt.alloc_buffer(data.len().max(1) * 2)?;
    b.write_bf16_bits(&tessl::tensor::f32_slice_to_bf16(data));
    Ok(b)
}

/// An f32 buffer of `n` NaNs, so an element a kernel never wrote is caught by
/// [`check_lane_output`].
pub fn poisoned_f32(rt: &std::sync::Arc<tessl::GpuRuntime>, n: usize) -> Result<tessl::tensor::GpuBuffer, String> {
    upload_f32(rt, &vec![f32::NAN; n])
}

/// A bf16 buffer of `n` NaNs.
pub fn poisoned_bf16(rt: &std::sync::Arc<tessl::GpuRuntime>, n: usize) -> Result<tessl::tensor::GpuBuffer, String> {
    let b = rt.alloc_buffer(n.max(1) * 2)?;
    b.write_bf16_bits(&vec![0x7fc0u16; n.max(1)]);
    Ok(b)
}

/// The first `n` elements of a bf16 buffer, widened (after a `synchronize`).
pub fn read_bf16(rt: &tessl::GpuRuntime, b: &tessl::tensor::GpuBuffer, n: usize) -> Result<Vec<f32>, String> {
    rt.synchronize()?;
    let bits = b.try_contents_u16()?;
    if bits.len() < n {
        return Err(format!("bf16 buffer holds {} elements, need {n}", bits.len()));
    }
    Ok(bits[..n].iter().map(|&x| tessl::tensor::bf16_bits_to_f32(x)).collect())
}

/// `x / sqrt(mean(x^2) + eps) * w` over one row in f64 (`* w`, not `* (1 +
/// w)`; no weight when `w` is `None`): the host reference of a row norm.
pub fn host_rms(x: &[f64], w: Option<&[f32]>, eps: f64) -> Vec<f64> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| v * inv * w.map_or(1.0, |w| f64::from(w[i])))
        .collect()
}

/// A lane's result must be every element finite and not all zero: a kernel
/// that wrote nothing (the buffer was poisoned with NaN first), or only zeros,
/// cannot post a time.
pub fn check_lane_output(lane: &str, out: &[f32]) -> Result<(), String> {
    if out.is_empty() {
        return Err(format!("{lane}: the output is empty"));
    }
    if let Some(i) = out.iter().position(|x| !x.is_finite()) {
        return Err(format!(
            "{lane}: element {i} of {} is {} (not finite)",
            out.len(),
            out[i]
        ));
    }
    if out.iter().all(|&x| x == 0.0) {
        return Err(format!("{lane}: all {} elements are zero", out.len()));
    }
    Ok(())
}

/// `got` against `want` (f64 host reference), each element within `abs + rel *
/// |want|`; names the worst element on failure.
pub fn check_against(lane: &str, got: &[f32], want: &[f64], abs: f64, rel: f64) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("{lane}: {} elements, expected {}", got.len(), want.len()));
    }
    let mut worst: Option<(usize, f64, f64)> = None;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let excess = (f64::from(g) - w).abs() - (abs + rel * w.abs());
        // A NaN element (`excess` NaN) is always a failure.
        if (excess.is_nan() || excess > 0.0) && worst.is_none_or(|(_, e, _)| excess.is_nan() || excess > e) {
            worst = Some((i, excess, f64::from(g)));
        }
    }
    match worst {
        None => Ok(()),
        Some((i, _, g)) => Err(format!(
            "{lane}: element {i} is {g}, host reference {} (tolerance {abs:e} + {rel:e} * |ref|)",
            want[i]
        )),
    }
}

/// Time `dispatch` (`warmup` untimed rounds, then `iters` timed ones, each
/// ended by a `synchronize`) and print `METRIC <lane>_ms <median>`. The caller
/// has already run and checked the kernel once.
pub fn time_lane(
    rt: &tessl::GpuRuntime,
    lane: &str,
    warmup: usize,
    iters: usize,
    mut dispatch: impl FnMut() -> Result<(), String>,
) -> Result<(), String> {
    let mut ms = Vec::with_capacity(iters);
    for round in 0..warmup + iters {
        let t0 = std::time::Instant::now();
        dispatch()?;
        rt.synchronize()?;
        if round >= warmup {
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    println!("METRIC {lane}_ms {:.4}", median(ms)?);
    Ok(())
}

/// The stdout line a `METRIC` lane ends with when it fails after it may have
/// printed metrics: `METRIC_INCOMPLETE reason`. A consumer that reads stdout
/// alone (a pipe, a saved log) must not take the `METRIC` lines before it as a
/// finished run. It does not start with `METRIC ` (with the space), so
/// `bench_paired`'s parser skips it; `bench_paired` itself refuses any lane
/// with a non-zero exit before it parses a line, so this adds nothing there.
pub fn incomplete_line(reason: &dyn std::fmt::Display) -> String {
    let one_line = reason.to_string().split_whitespace().collect::<Vec<_>>().join(" ");
    format!("METRIC_INCOMPLETE {one_line}")
}

/// Pass `result` through; on an error print [`incomplete_line`] to stdout.
/// Wrap only the part of a run that starts after every argument and
/// environment value has been validated.
pub fn mark_incomplete_on_error<T, E: std::fmt::Display>(result: Result<T, E>) -> Result<T, E> {
    if let Err(e) = &result {
        println!("{}", incomplete_line(e));
    }
    result
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

// ---------------------------------------------------------------------------
// Quiet-machine gating and JSON scalars for the profile binaries
// ---------------------------------------------------------------------------

/// Other tessl processes that may be using the GPU: test and bench binaries
/// under a `tessl*` checkout's `target/`, other than this one. Other sessions
/// share this machine; a sample taken while one runs is flagged, not trusted.
/// A GPU user this does not see (a display, a browser) stays in the baseline,
/// which [`gpu_utilization`] records.
pub fn others_running() -> Vec<String> {
    /// `PROC_ALL_PIDS` of `<sys/proc_info.h>`; the `libc` crate does not export it.
    const PROC_ALL_PIDS: u32 = 1;
    let me = std::process::id() as libc::pid_t;
    // SAFETY: with a null buffer `proc_listpids` only returns the bytes needed.
    let need = unsafe { libc::proc_listpids(PROC_ALL_PIDS, 0, std::ptr::null_mut(), 0) };
    if need <= 0 {
        return vec!["proc_listpids failed".to_string()];
    }
    let mut pids = vec![0 as libc::pid_t; need as usize / std::mem::size_of::<libc::pid_t>() + 64];
    let bytes = (pids.len() * std::mem::size_of::<libc::pid_t>()) as libc::c_int;
    // SAFETY: `pids` is a live buffer of `bytes` bytes.
    let got = unsafe { libc::proc_listpids(PROC_ALL_PIDS, 0, pids.as_mut_ptr().cast(), bytes) };
    if got <= 0 {
        return vec!["proc_listpids failed".to_string()];
    }
    let n = got as usize / std::mem::size_of::<libc::pid_t>();
    let mut found = Vec::new();
    for &pid in &pids[..n.min(pids.len())] {
        if pid <= 0 || pid == me {
            continue;
        }
        let mut buf = [0u8; 4096];
        // SAFETY: `buf` is a live buffer of the length passed.
        let len = unsafe { libc::proc_pidpath(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
        if len <= 0 {
            continue;
        }
        let path = String::from_utf8_lossy(&buf[..len as usize]).into_owned();
        let in_target =
            path.contains("/research/tessl") && (path.contains("/target/release/") || path.contains("/target/debug/"));
        let file = path.rsplit('/').next().unwrap_or("");
        let gpu_bin = path.contains("/deps/")
            || file.starts_with("bench_")
            || file.starts_with("probe_")
            || file.starts_with("profile_");
        if in_target && gpu_bin {
            found.push(file.to_string());
        }
    }
    found
}

/// Milliseconds this process has spent waiting for a quiet machine.
static WAITED_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Wait for [`others_running`] to be empty, for at most `PROFILE_WAIT_S`
/// seconds (default 1800) in total over the whole run; false if it was not
/// empty when the budget ran out.
pub fn wait_quiet() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let budget_ms = 1000
        * std::env::var("PROFILE_WAIT_S")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1800);
    while !others_running().is_empty() {
        if WAITED_MS.load(Relaxed) >= budget_ms {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(1000));
        WAITED_MS.fetch_add(1000, Relaxed);
    }
    true
}

/// `run` once the machine is quiet, retaken (up to 3 times) when another tessl
/// GPU process appears during it: (result, contended, retakes).
pub fn gated<T>(mut run: impl FnMut() -> Result<T, String>) -> Result<(T, bool, usize), String> {
    let mut retakes = 0;
    loop {
        let quiet = wait_quiet();
        let r = run()?;
        let seen_after = !others_running().is_empty();
        if !seen_after && quiet {
            return Ok((r, false, retakes));
        }
        if retakes >= 3 || !quiet {
            return Ok((r, true, retakes));
        }
        retakes += 1;
    }
}

/// The samples of `v` taken while the machine was quiet, or all of them when
/// none was.
pub fn clean_or_all(v: &[f64], contended: &[bool]) -> Vec<f64> {
    let clean: Vec<f64> = v.iter().zip(contended).filter(|(_, &c)| !c).map(|(&x, _)| x).collect();
    if clean.is_empty() {
        v.to_vec()
    } else {
        clean
    }
}

/// The device's GPU utilization as `ioreg` reports it, for the machine-state
/// record (other sessions share this GPU).
pub fn gpu_utilization() -> String {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.split("\"Device Utilization %\"=")
                .nth(1)
                .and_then(|t| t.split(|c: char| !c.is_ascii_digit()).next())
                .unwrap_or("unknown")
                .to_string()
        }
        Err(e) => format!("ioreg failed: {e}"),
    }
}

pub fn min_of(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

pub fn median_of(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

pub fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.6}")
    } else {
        "null".to_string()
    }
}

pub fn jarr(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|&x| jnum(x)).collect::<Vec<_>>().join(","))
}

pub fn jbools(v: &[bool]) -> String {
    format!("[{}]", v.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","))
}

pub fn jstr(s: &str) -> String {
    format!("{s:?}")
}

#[cfg(test)]
mod tests {
    use super::{
        fill, fill_affine, incomplete_line, median, parse_count, ratio_stats, round_order, run_paired, self_spread, Arm,
    };

    #[test]
    fn incomplete_marker_is_one_line_and_not_a_metric() {
        let l = incomplete_line(&"gpu failed\nsecond line");
        assert_eq!(l, "METRIC_INCOMPLETE gpu failed second line");
        assert!(!l.starts_with("METRIC "));
    }

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
    fn fill_is_the_seeded_uniform_stream_every_bench_used() {
        let v = fill(4096, 7);
        assert_eq!(v, fill(4096, 7));
        assert_ne!(v, fill(4096, 8));
        assert!(v.iter().all(|x| (-1.0..=1.0).contains(x)));
        assert!(v.iter().any(|x| *x < -0.5) && v.iter().any(|x| *x > 0.5));
        // The retired per-binary copies: one LCG step, high 32 bits, mapped to [-1, 1].
        let mut s = 7u64.wrapping_mul(6364136223846793005).wrapping_add(1);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let first = (((s >> 32) as u32) as f64 / (u32::MAX as f64) * 2.0 - 1.0) as f32;
        assert_eq!(v[0], first);
        let a = fill_affine(16, 7, 0.5, 2.0);
        assert!(a.iter().zip(fill(16, 7)).all(|(y, x)| *y == x * 0.5 + 2.0));
    }

    #[test]
    fn benchmark_count_parser_is_strict_and_bounded() {
        assert_eq!(parse_count("N", "7", 1).unwrap(), 7);
        assert_eq!(parse_count("N", "0", 0).unwrap(), 0);
        for bad in [
            "",
            " 7",
            "7 ",
            " 7 ",
            "+7",
            "-1",
            "007",
            "00",
            "0x7",
            "1e3",
            "1_0",
            "\u{661}",
            "not-a-count",
            "7\n",
            "1000000",
            "18446744073709551615",
            "18446744073709551616",
        ] {
            assert!(parse_count("N", bad, 0).is_err(), "{bad:?} accepted");
        }
        assert!(parse_count("N", "0", 1).unwrap_err().contains("minimum"));
        assert!(parse_count("N", "999999", 1).is_ok());
        assert!(parse_count("N", "1000000", 1).unwrap_err().contains("maximum"));
    }
}
