//! GEMM shape sweep: TensorOps vs simdgroup, JSON out for cross-runtime compare.
//!
//! Companion to `bench/gemm_sweep_mlx.py` (MLX + PyTorch MPS lanes). Protocol is
//! pinned to match: same shapes, same warmup/iters, **synchronize every iteration**
//! so no lane hides dispatch cost behind pipelining, median over iters.
//!
//! `--dump-parity DIR` is a separate job from the timing sweep: it runs every
//! lane once per seed on `PARITY_SHAPE` and writes A/B/C as .npy for the numeric
//! check in the Python lane. `BENCH_PARITY_SEEDS` sets the draw count.

mod common;

use common::{env_usize, fill_dist, median, Dist};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tessl::gemm::{cast_f32_to_bf16, gemm, GemmBackend};
use tessl::npy::write_npy_f32;
use tessl::runtime::GpuRuntime;
use tessl::tensor::Tensor;

/// (M, N, K, label). Square ladder + the projection shapes arch_02 actually runs.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (512, 512, 512, "square_512"),
    (1024, 1024, 1024, "square_1024"),
    (2048, 2048, 2048, "square_2048"),
    (4096, 4096, 4096, "square_4096"),
    (2048, 768, 768, "qkv_proj"),
    (8192, 3072, 768, "mlp_up"),
    (8192, 768, 3072, "mlp_down"),
    (4096, 4096, 1024, "tall_k1024"),
];

/// The shape `--dump-parity` scores when `BENCH_PARITY_SHAPE` is unset.
///
/// One shape per invocation rather than the whole ladder is deliberate: the
/// ladder at eight draws is ~12 GB of .npy, so `bench/parity_ladder.py` walks
/// it one label at a time and deletes as it goes. Forward error grows with K,
/// so any single shape is a floor across the ladder rather than a bound.
const DEFAULT_PARITY_SHAPE: &str = "square_1024";

/// Default operand draws for the parity dump. One draw cannot separate a real
/// error bound from a lucky sample; the Python side reports the worst across
/// all of them rather than the last.
const DEFAULT_PARITY_SEEDS: usize = 8;

/// A shape resolved at runtime: (M, N, K, label).
type Shape = (usize, usize, usize, String);

/// One measured or scored lane: (name, backend, A operand, B operand).
type Lane = (String, GemmBackend, Tensor, Tensor);

/// Uniform fill, for the timing sweep — which does not vary the distribution
/// because GEMM throughput does not depend on operand values.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    fill_dist(n, seed, Dist::Uniform)
}

fn time_backend(
    rt: &GpuRuntime,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
    warmup: usize,
    iters: usize,
) -> Result<Vec<f64>, String> {
    for _ in 0..warmup {
        gemm(a, b, c, backend)?;
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        gemm(a, b, c, backend)?;
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(samples)
}

/// `BENCH_SHAPES="MxNxK,MxNxK,..."` overrides the built-in ladder so this lane
/// and the Python lane can be pointed at an identical diagnostic grid.
///
/// Every rejection names the offending entry. The previous `parse().unwrap()`
/// reported a malformed grid as `called Option::unwrap() on a None value`.
fn shapes_from_env() -> Result<Option<Vec<Shape>>, String> {
    let raw = match std::env::var("BENCH_SHAPES") {
        Ok(v) => v,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(e) => return Err(format!("BENCH_SHAPES: {e}")),
    };
    let mut out = Vec::new();
    for spec in raw.split(',').filter(|s| !s.trim().is_empty()) {
        let spec = spec.trim();
        let parts: Vec<&str> = spec.split('x').collect();
        if parts.len() != 3 {
            return Err(format!("BENCH_SHAPES entry must be MxNxK, got {spec:?}"));
        }
        let mut dims = [0usize; 3];
        for (slot, text) in dims.iter_mut().zip(parts.iter()) {
            *slot = text
                .trim()
                .parse()
                .map_err(|_| format!("BENCH_SHAPES entry {spec:?}: {text:?} is not an integer"))?;
            if *slot == 0 {
                return Err(format!("BENCH_SHAPES entry {spec:?} has a zero dimension"));
            }
        }
        let (m, n, k) = (dims[0], dims[1], dims[2]);
        out.push((m, n, k, format!("{m}x{n}x{k}")));
    }
    if out.is_empty() {
        return Err("BENCH_SHAPES is set but parsed to no shapes".to_string());
    }
    Ok(Some(out))
}

/// Resolve `BENCH_PARITY_SHAPE` against the ladder.
///
/// An unknown label lists the valid ones rather than falling back to the
/// default: silently scoring a shape other than the one asked for is the same
/// defect as silently scoring fewer lanes than asked for.
fn parity_shape_from_env() -> Result<(usize, usize, usize, &'static str), String> {
    let want = match std::env::var("BENCH_PARITY_SHAPE") {
        Ok(v) => v.trim().to_string(),
        Err(std::env::VarError::NotPresent) => DEFAULT_PARITY_SHAPE.to_string(),
        Err(e) => return Err(format!("BENCH_PARITY_SHAPE: {e}")),
    };
    SHAPES
        .iter()
        .find(|(.., l)| *l == want)
        .copied()
        .ok_or_else(|| {
            let all: Vec<&str> = SHAPES.iter().map(|(.., l)| *l).collect();
            format!("BENCH_PARITY_SHAPE={want:?} is not a ladder label; expected one of {all:?}")
        })
}

/// tf32 is a runtime mode rather than a dtype, so the lane name selects it.
/// Both callers route through here and both clear it afterwards — leaving it
/// latched leaked relaxed precision into whatever ran next.
fn relaxed_for(lane: &str) -> bool {
    lane == "tensorops-tf32"
}

/// The lane list has one construction site so the parity dump and the timing
/// sweep can never disagree about what "every lane" means.
///
/// Must be rebuilt whenever A or B change: `cast_f32_to_bf16` snapshots its
/// source into a fresh tensor, so a list built once and reused across operand
/// draws would keep serving the first draw's bf16 operands while the f32 lanes
/// moved on.
fn build_lanes(
    rt: &GpuRuntime,
    a: &Tensor,
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    backends: &[(&str, GemmBackend)],
) -> Result<Vec<Lane>, String> {
    let mut lanes: Vec<Lane> = Vec::new();
    for &(bname, backend) in backends {
        lanes.push((
            format!("{bname}-f32"),
            backend,
            a.view(&[m, k], 0),
            b.view(&[k, n], 0),
        ));
    }
    if rt.has_tensorops() {
        // bf16 operands, f32 accumulate — the path `gemm_train` takes under
        // PrecisionMode::Bf16.
        lanes.push((
            "tensorops-bf16".to_string(),
            GemmBackend::TensorOps,
            cast_f32_to_bf16(a)?,
            cast_f32_to_bf16(b)?,
        ));
        // tf32-class relaxed precision on f32 operands (opt-in --tf32 path).
        // Without this lane the relaxed-f32 kernels never appear in any
        // cross-runtime comparison, because a sweep that only toggles dtype
        // cannot see a runtime mode.
        lanes.push((
            "tensorops-tf32".to_string(),
            GemmBackend::TensorOps,
            a.view(&[m, k], 0),
            b.view(&[k, n], 0),
        ));
    }
    Ok(lanes)
}

/// Numeric-parity dump: every lane, every seed, one shape.
///
/// This is deliberately not folded into the timing loop. When it was, the
/// dumped C was whatever the last timed iteration happened to leave in the
/// buffer — under `BENCH_ITERS=0 BENCH_WARMUP=0` that was a buffer no gemm had
/// ever written, dumped and scored as though it were a result. The dump also
/// used to be filtered to `-f32` lane names, which left tensorops-bf16 and
/// tensorops-tf32 — the two lanes carrying the headline speed ratios — out of
/// the only numeric check that scores tessl and the runtimes it is compared
/// against on one reference.
fn dump_parity(
    // `alloc_tensor_f32` takes `self: &Arc<Self>`, so this cannot narrow to
    // `&GpuRuntime` the way the other helpers do.
    rt: &Arc<GpuRuntime>,
    dir: &str,
    shape: (usize, usize, usize, &str),
    dist: Dist,
    seeds: usize,
    backends: &[(&str, GemmBackend)],
) -> Result<(), String> {
    let (m, n, k, label) = shape;

    let d = Path::new(dir);
    std::fs::create_dir_all(d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;

    let a = rt.alloc_tensor_f32(&[m, k])?;
    let b = rt.alloc_tensor_f32(&[k, n])?;
    let c = rt.alloc_tensor_f32(&[m, n])?;
    // A kernel that writes nothing must not inherit the previous lane's result
    // and score as if it had run. Pre-zeroing turns that into ~1.0 relative
    // error, which is loud, instead of a clean pass.
    let zero = vec![0f32; m * n];

    let mut lane_names: Vec<String> = Vec::new();
    let mut result = Ok(());
    'seeds: for s in 0..seeds {
        let sd = d.join(format!("seed_{s:02}"));
        if let Err(e) = std::fs::create_dir_all(&sd) {
            result = Err(format!("mkdir {}: {e}", sd.display()));
            break 'seeds;
        }
        // Seed 0 is the historical (1, 2) draw, so the single-seed number this
        // replaces stays reproducible as seed_00.
        let a_host = fill_dist(m * k, 2 * s as u64 + 1, dist);
        let b_host = fill_dist(k * n, 2 * s as u64 + 2, dist);
        // The scorer rejects non-finite operands; catching it here names the
        // distribution instead of blaming the dump.
        if let Some(bad) = a_host.iter().chain(&b_host).find(|v| !v.is_finite()) {
            rt.set_relaxed_precision(false);
            return Err(format!(
                "{} seed {s}: operand {bad} is not finite",
                dist.name()
            ));
        }
        a.buffer.write_f32(&a_host);
        b.buffer.write_f32(&b_host);

        let lanes = match build_lanes(rt, &a, &b, m, n, k, backends) {
            Ok(v) => v,
            Err(e) => {
                result = Err(e);
                break 'seeds;
            }
        };
        if s == 0 {
            lane_names = lanes.iter().map(|(l, ..)| l.clone()).collect();
        } else {
            // Cheap, and it is the assumption the manifest encodes: one lane
            // list describes every seed directory.
            let now: Vec<String> = lanes.iter().map(|(l, ..)| l.clone()).collect();
            if now != lane_names {
                result = Err(format!(
                    "lane set changed between seeds: {lane_names:?} then {now:?}"
                ));
                break 'seeds;
            }
        }

        if let Err(e) = write_npy_f32(&sd.join("parity_a.npy"), &[m, k], &a_host)
            .and_then(|()| write_npy_f32(&sd.join("parity_b.npy"), &[k, n], &b_host))
        {
            result = Err(e);
            break 'seeds;
        }

        for (lane, backend, la, lb) in &lanes {
            rt.set_relaxed_precision(relaxed_for(lane));
            c.buffer.write_f32(&zero);
            if let Err(e) = gemm(la, lb, &c, *backend).and_then(|()| rt.synchronize()) {
                result = Err(format!("lane {lane} seed {s}: {e}"));
                break 'seeds;
            }
            let out = c.buffer.read_f32()[..m * n].to_vec();
            if let Some(i) = out.iter().position(|v| !v.is_finite()) {
                result = Err(format!(
                    "lane {lane} seed {s}: non-finite output {} at element {i} — \
                     refusing to dump a result the scorer would report as a number",
                    out[i]
                ));
                break 'seeds;
            }
            if let Err(e) = write_npy_f32(&sd.join(format!("parity_c_{lane}.npy")), &[m, n], &out) {
                result = Err(e);
                break 'seeds;
            }
        }
        eprintln!(
            "parity {label}/{} seed {s} / {seeds}: {} lanes dumped",
            dist.name(),
            lanes.len()
        );
    }
    // Cleared on every exit path, success or not. This is process-global GPU
    // state and a latched `true` is invisible at the next call site.
    rt.set_relaxed_precision(false);
    result?;

    // Written last, and only on success, so an interrupted dump leaves no
    // manifest and the scorer refuses it outright rather than reporting a
    // partial run as a complete one.
    let lanes_json: Vec<String> = lane_names.iter().map(|l| format!("{l:?}")).collect();
    let seeds_json: Vec<String> = (0..seeds).map(|s| format!("\"seed_{s:02}\"")).collect();
    std::fs::write(
        d.join("parity_manifest.json"),
        format!(
            r#"{{"shape":"{label}","dist":"{}","m":{m},"n":{n},"k":{k},"lanes":[{}],"seeds":[{}]}}"#,
            dist.name(),
            lanes_json.join(","),
            seeds_json.join(",")
        ),
    )
    .map_err(|e| format!("write manifest: {e}"))?;
    eprintln!(
        "parity dump complete: {} lanes x {seeds} seeds at {label}/{} -> {}",
        lane_names.len(),
        dist.name(),
        d.display()
    );
    Ok(())
}

/// Emit the kernel trace for `bench/kernel_coverage.py`. Prints nothing when
/// tracing is off, so normal runs are unchanged.
fn emit_kernel_trace() {
    if tessl::runtime::kernel_trace_enabled() {
        eprintln!(
            "KERNEL_TRACE {}",
            tessl::runtime::traced_kernels().join(",")
        );
    }
}

fn run() -> Result<(), String> {
    let rt = GpuRuntime::new()?;

    let args: Vec<String> = std::env::args().collect();
    let dump_dir = match args.iter().position(|a| a == "--dump-parity") {
        Some(i) => Some(
            args.get(i + 1)
                .filter(|v| !v.starts_with("--"))
                .ok_or("--dump-parity requires a directory argument")?
                .clone(),
        ),
        None => None,
    };

    let mut backends = vec![("simdgroup", GemmBackend::Simdgroup)];
    if rt.has_tensorops() {
        backends.insert(0, ("tensorops", GemmBackend::TensorOps));
    } else {
        eprintln!("warning: TensorOps absent from metallib; simdgroup lane only");
    }

    if let Some(dir) = dump_dir {
        // BENCH_SHAPES relabels every shape, so it never matched the parity
        // label and the dump quietly produced nothing at all. Rejecting the
        // pair is the only outcome that cannot be mistaken for a check that ran.
        if std::env::var_os("BENCH_SHAPES").is_some() {
            return Err(
                "BENCH_SHAPES does not apply to --dump-parity; select the parity \
                        shape with BENCH_PARITY_SHAPE (a ladder label) rather than \
                        have BENCH_SHAPES silently ignored"
                    .to_string(),
            );
        }
        let shape = parity_shape_from_env()?;
        let dist = match std::env::var("BENCH_PARITY_DIST") {
            Ok(v) => Dist::parse(v.trim())?,
            Err(std::env::VarError::NotPresent) => Dist::Uniform,
            Err(e) => return Err(format!("BENCH_PARITY_DIST: {e}")),
        };
        let seeds = env_usize("BENCH_PARITY_SEEDS", DEFAULT_PARITY_SEEDS, 1)?;
        return dump_parity(&rt, &dir, shape, dist, seeds, &backends);
    }

    let warmup = env_usize("BENCH_WARMUP", 10, 0)?;
    let iters = env_usize("BENCH_ITERS", 50, 1)?;
    let shapes: Vec<Shape> = match shapes_from_env()? {
        Some(v) => v,
        None => SHAPES
            .iter()
            .map(|&(m, n, k, l)| (m, n, k, l.to_string()))
            .collect(),
    };

    let mut rows: Vec<String> = Vec::new();
    for (m, n, k, label) in shapes.iter().map(|(m, n, k, l)| (*m, *n, *k, l.as_str())) {
        let a = rt.alloc_tensor_f32(&[m, k])?;
        let b = rt.alloc_tensor_f32(&[k, n])?;
        let c = rt.alloc_tensor_f32(&[m, n])?;
        a.buffer.write_f32(&fill(m * k, 1));
        b.buffer.write_f32(&fill(k * n, 2));

        // 2*M*N*K FLOP per GEMM.
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let lanes = build_lanes(&rt, &a, &b, m, n, k, &backends)?;

        for (bname, backend, la, lb) in &lanes {
            let (bname, backend) = (bname.as_str(), *backend);
            rt.set_relaxed_precision(relaxed_for(bname));
            let samples = time_backend(&rt, la, lb, &c, backend, warmup, iters)?;
            let med = median(samples.clone())?;
            if med <= 0.0 {
                rt.set_relaxed_precision(false);
                return Err(format!(
                    "{label}/{bname}: median {med} ms is not positive; the timer \
                     resolved nothing and GFLOP/s would be infinite"
                ));
            }
            let best = samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let gflops = flop / (med * 1e6);
            eprintln!(
                "{label:<12} {bname:<10} M={m} N={n} K={k}  {med:8.3} ms  {gflops:8.1} GFLOP/s"
            );
            rows.push(format!(
                r#"{{"shape":"{label}","backend":"{bname}","runtime":"metal-native","m":{m},"n":{n},"k":{k},"median_ms":{med:.6},"best_ms":{best:.6},"gflops":{gflops:.3}}}"#
            ));
        }
    }
    rt.set_relaxed_precision(false);
    println!("[{}]", rows.join(","));
    Ok(())
}

fn main() -> Result<(), String> {
    // Wrapped rather than called before each `return`: --dump-parity exits
    // early and an error exits earlier still, and a trace that silently skips
    // those paths would under-report coverage exactly where it matters.
    let outcome = run();
    emit_kernel_trace();
    outcome
}
